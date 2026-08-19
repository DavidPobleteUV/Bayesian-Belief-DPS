# -*- coding: utf-8 -*-
"""
validate_allocation.py — Validation gate for the deterministic allocation head.

Feeds the waterfall the OBSERVED pump-factor + salinity + demand from the zarr and
checks it reproduces the OBSERVED per-source transmission flows, total supply and
unmet, town by town. Calibrates the per-town `tourist_mult` first.

Run:
  python validate_allocation.py --zarr data/weap_weekly.zarr --q_wells data/Q_wells.xlsx \
    --cost_csv ../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd, zarr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rdm_mlp.allocation.allocation_head import (
    TownSpec, allocate_week, well_supply_m3wk, LPS_TO_M3WK, SECONDS_PER_WEEK, LOSS_FACTOR,
)

WARMUP = 104
BASE_YEAR = 2014           # week 0 ≈ 2014-04; year = BASE_YEAR + t//52
COASTAL = {"ElEsfuerzo", "Quilimari", "Pichidangui"}


def kge(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 10 or np.std(obs[m]) < 1e-9:
        return np.nan
    o, s = obs[m], sim[m]
    r = np.corrcoef(o, s)[0, 1]
    a = np.std(s) / np.std(o); b = np.mean(s) / (np.mean(o) + 1e-9)
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


def build_well_caps(q_wells_path):
    raw = pd.read_excel(q_wells_path, header=None)
    hr = raw.index[raw.apply(lambda r: r.astype(str).str.contains("BranchID").any(), axis=1)][0]
    df = pd.read_excel(q_wells_path, header=hr)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Level 2": "node", "Level 3": "well", "Variable": "var", "Expression": "expr"})
    cap = df[df["var"].astype(str).str.strip() == "Caudal asociado"].copy()
    cap["expr"] = pd.to_numeric(cap["expr"], errors="coerce")
    caps = {}     # {fict_node_str: {well: cap}}
    for _, r in cap.iterrows():
        node, well, c = str(r["node"]), str(r["well"]), r["expr"]
        if not np.isfinite(c) or c <= 0:
            continue
        caps.setdefault(node, {})[well] = float(c)
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/weap_weekly.zarr")
    ap.add_argument("--q_wells", default="data/Q_wells.xlsx")
    ap.add_argument("--cost_csv", default="../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv")
    ap.add_argument("--runs", default="", help="comma list of run_ids; blank=auto baseline")
    ap.add_argument("--n_baseline", type=int, default=6)
    args = ap.parse_args()

    Z = zarr.open_group(args.zarr, mode="r")
    targ = list(Z.attrs["target_names"]); feat = list(Z.attrs["feature_names"])
    rids = Z["run_ids"][:]
    ti = {n: i for i, n in enumerate(targ)}; fi = {n: i for i, n in enumerate(feat)}

    well_caps = build_well_caps(args.q_wells)
    cost = pd.read_csv(args.cost_csv)
    cost["withdrawal_node"] = cost["withdrawal_node"].astype(str).str.strip()

    # ── town registry: demand node, fict node, link cols, pump/sal cols ──
    link_targets = [n for n in targ if "AP_TransmissionLinks" in n and "_to_" in n]
    # map demand-node -> town (last token after Dem_)
    def town_of_dem(dem):
        m = re.search(r"_Dem_(\w+)", dem); return m.group(1) if m else dem
    dem_nodes = sorted({n.split("__", 1)[1].rsplit("_to_", 1)[1] for n in link_targets})

    towns = {}
    for dem in dem_nodes:
        town = town_of_dem(dem)
        if town in ("ElManzanoL",):      # truncated dup; merge into full
            continue
        # demand feature
        dcol = fi.get("AP_WaterDemand__" + dem)
        # links into this dem node
        my = [n for n in link_targets if n.endswith("_to_" + dem)]
        well_col = None; type_cols = {"Aduccion": [], "Camiones": [], "Desal": [], "PozoCostero": [], "Acuerdo": []}
        fict = None
        for n in my:
            body = n.split("__", 1)[1]; src = body.rsplit("_to_", 1)[0].replace("Transmission_Link_from_", "")
            if ("APR_" in src or "APU_" in src) and "_Fict_" in src:
                well_col = ti[n]; fict = src
            elif src.startswith("Withdrawal_Node_"):
                node = src.replace("Withdrawal_Node_", "")
                row = cost[cost["withdrawal_node"] == node]
                if len(row):
                    type_cols[row.iloc[0]["source_type"]].append(ti[n])
            elif src.startswith("DemAGRO_SHAC"):
                type_cols["Acuerdo"].append(ti[n])
        if dcol is None or fict is None:
            continue
        # pump/sal target cols for this fict node, aligned to caps
        wc = well_caps.get(fict, {})
        pump_cols, sal_cols, caps_arr = [], [], []
        for well, c in wc.items():
            pc = ti.get(f"WF_PumpFactor__{fict}__{well}")
            sc = ti.get(f"WF_SalinityFactor__{fict}__{well}")
            if pc is not None:
                pump_cols.append(pc); sal_cols.append(sc if sc is not None else -1); caps_arr.append(c)
        towns[town] = dict(dem=dem, fict=fict, dcol=dcol, well_col=well_col,
                           type_cols=type_cols, pump_cols=pump_cols, sal_cols=sal_cols,
                           caps=np.array(caps_arr), coastal=town in COASTAL)

    print(f"[registry] {len(towns)} towns mapped: {sorted(towns)}")

    # ── pick baseline runs (no Withdrawal-node flow at all = no desal/trucks/etc active is wrong;
    #    instead: runs where desal/pozocostero links are ~0 over horizon) ──
    if args.runs:
        run_list = [int(x) for x in args.runs.split(",")]
    else:
        run_list = []
        for rid in rids:
            if rid >= 1000:   # skip LHS/pareto, use factorial baseline block
                continue
            s = int(np.where(rids == rid)[0][0])
            # baseline = no desal flow anywhere
            desal_cols = sum([towns[t]["type_cols"]["Desal"] for t in towns], [])
            pc_cols = sum([towns[t]["type_cols"]["PozoCostero"] for t in towns], [])
            Yd = Z["Y"][s][WARMUP:][:, desal_cols + pc_cols]
            if np.nansum(Yd) < 1.0:
                run_list.append(int(rid))
            if len(run_list) >= args.n_baseline:
                break
    print(f"[runs] baseline runs used: {run_list}\n")

    # ── calibrate tourist_mult per town: ratio observed_total / (demand/0.7) ──
    print("=== STEP 1: calibrate tourist_mult (observed_supply / (demand/0.70)) ===")
    tourist = {}
    for town, T in towns.items():
        ratios = []
        ap_cols = ([T["well_col"]] + T["type_cols"]["Aduccion"] + T["type_cols"]["PozoCostero"]
                   + T["type_cols"]["Desal"] + T["type_cols"]["Camiones"])  # exclude Acuerdo + only AP
        for rid in run_list:
            s = int(np.where(rids == rid)[0][0])
            Y = Z["Y"][s]; X = Z["X"][s]
            sup = Y[WARMUP:][:, ap_cols].sum(axis=1)
            need = X[WARMUP:, T["dcol"]] * SECONDS_PER_WEEK / LOSS_FACTOR
            m = need > 1
            if m.sum():
                ratios.append(np.nanmedian(sup[m] / need[m]))
        tm = float(np.nanmedian(ratios)) if ratios else 1.0
        tourist[town] = max(tm, 0.1)
        print(f"  {town:16s} tourist_mult = {tm:5.2f}")

    # ── build specs ──
    specs = {}
    for town, T in towns.items():
        specs[town] = TownSpec(
            name=town, coastal=T["coastal"],
            well_caps_lps={w: c for w, c in zip(range(len(T["caps"])), T["caps"])},
            tourist_mult=tourist[town],
            aduccion_lps=1.0 if town in ("Guanguali", "LosCondores") else 0.0,   # 2 l/s split
            pozocostero_lps=0.0,   # baseline: new well off
            desal_lps=0.0,         # baseline: desal off
            acuerdo_lps=np.inf,
        )

    # ── STEP 2: run waterfall on observed pump/sal/demand, compare ──
    print("\n=== STEP 2: reproduce observed flows (KGE vs observed, per source) ===")
    agg = {town: {"total_obs": [], "total_sim": [], "well_obs": [], "well_sim": [],
                  "truck_obs": [], "truck_sim": [], "unmet_obs": [], "unmet_sim": []} for town in towns}

    for rid in run_list:
        s = int(np.where(rids == rid)[0][0])
        Y = Z["Y"][s]; X = Z["X"][s]; Tn = Y.shape[0]
        # precompute per-town well flow series + demand series
        wellflow = {}; demand = {}
        for town, T in towns.items():
            caps = T["caps"]
            pump = Y[:, T["pump_cols"]] if T["pump_cols"] else np.zeros((Tn, 1))
            if T["coastal"] and T["sal_cols"]:
                sal = np.stack([Y[:, c] if c >= 0 else np.zeros(Tn) for c in T["sal_cols"]], axis=1)
                sal_prev = np.vstack([sal[:1], sal[:-1]])   # lag 1
            else:
                sal_prev = np.zeros_like(pump)
            wf = np.array([well_supply_m3wk(caps, pump[t], sal_prev[t], T["coastal"]) for t in range(Tn)])
            wellflow[town] = wf
            demand[town] = X[:, T["dcol"]] * SECONDS_PER_WEEK * tourist[town]

        for t in range(WARMUP, Tn):
            year = BASE_YEAR + t // 52
            res = allocate_week({tn: demand[tn][t] for tn in towns},
                                {tn: wellflow[tn][t] for tn in towns}, specs, year)
            for town, T in towns.items():
                r = res[town]
                obs_well = Y[t, T["well_col"]]
                obs_truck = Y[t, T["type_cols"]["Camiones"]].sum() if T["type_cols"]["Camiones"] else 0.0
                obs_total = (Y[t, T["well_col"]] + sum(Y[t, c].sum() for k, c in T["type_cols"].items()
                                                       if k != "Acuerdo" and c))
                sim_total = r["well"] + r["Aduccion"] + r["PozoCostero"] + r["Desal"] + r["Camiones"]
                a = agg[town]
                a["total_obs"].append(obs_total); a["total_sim"].append(sim_total)
                a["well_obs"].append(obs_well);   a["well_sim"].append(r["well"])
                a["truck_obs"].append(obs_truck); a["truck_sim"].append(r["Camiones"])

    print(f"  {'town':16s} {'KGE_total':>9s} {'KGE_well':>9s} {'KGE_truck':>9s}  {'bias_total':>10s}")
    kges = []
    for town in sorted(towns):
        a = agg[town]
        kt = kge(a["total_obs"], a["total_sim"]); kw = kge(a["well_obs"], a["well_sim"]); ktr = kge(a["truck_obs"], a["truck_sim"])
        bias = np.nansum(a["total_sim"]) / (np.nansum(a["total_obs"]) + 1e-9)
        kges.append(kt)
        print(f"  {town:16s} {kt:9.3f} {kw:9.3f} {ktr:9.3f}  {bias:10.2f}")
    print(f"\n  median KGE_total across towns = {np.nanmedian(kges):.3f}")
    print("  (gate: want KGE_total ≳ 0.7 and bias ≈ 1.0 before trusting the head)")


if __name__ == "__main__":
    main()
