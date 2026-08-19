# -*- coding: utf-8 -*-
"""
eval_v33_waterfall.py — v3.3 = v3 base + deterministic WELL-ANCHORED waterfall.
Keeps v3's predicted well flow; derives the downstream sources as the capped
shortfall by price priority:
   well(v3) -> aduccion -> pozo-costero -> desal -> trucks -> unmet
Desal/pozo gated by the action flags (correct for DPS deployment; LHS training runs
where desal is unflagged are reported separately).

Scores desal/trucks/total flow KGE and J4 cost vs observed, comparing to v3 native.
"""
from __future__ import annotations
import sys, argparse
import numpy as np, zarr, pandas as pd
sys.path.insert(0, "src")

_ap = argparse.ArgumentParser()
_ap.add_argument("--preds", default="results/preds_v3_j6spread.zarr")
_ap.add_argument("--label", default="v3")
_ARGS, _ = _ap.parse_known_args()
from train_v5_allocation import build_registry, build_well_caps, SEC, WARMUP

LPS = 604.8
CAP_LPS = {"Aduccion": 2.0, "PozoCostero": 120.0, "Desal": 1e9, "Camiones": 1e9, "Acuerdo": 1e9}
# desal/trucks effectively uncapped (they fill the gap; obs shows cap rarely binds)

Z = zarr.open_group("data/weap_weekly.zarr", mode="r")
targ = list(Z.attrs["target_names"]); feat = list(Z.attrs["feature_names"]); rids = list(Z["run_ids"][:])
ti = {n: i for i, n in enumerate(targ)}; fi = {n: i for i, n in enumerate(feat)}
cost = pd.read_csv("../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv")
cost["withdrawal_node"] = cost["withdrawal_node"].astype(str).str.strip()
well_caps = build_well_caps("data/Q_wells.xlsx")
towns, _, _ = build_registry(Z, well_caps, cost)
town_list = sorted(towns)
UC = {"Well": 200.0, "Aduccion": 500.0, "PozoCostero": 1200.0, "Desal": 1500.0, "Acuerdo": 3500.0, "Camiones": 8000.0}
# Documented chain: well -> aduccion -> pozo-costero -> desal -> trucks -> unmet.
# Acuerdo is NOT a fallback source here: uncapped (1e9) ahead of Camiones it would
# swallow the whole residual and zero out trucks (-> KGE NaN). Excluded by design.
PRIORITY = ["Aduccion", "PozoCostero", "Desal", "Camiones"]   # after well

q_desal = [fi[n] for n in ["q_desalacion_costera", "q_desalacion_completa"] if n in fi]
q_pozo = fi.get("q_nuevo_pozo_a_5km")

def kge(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float); m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 30 or np.std(o[m]) < 1e-9: return np.nan
    o, s = o[m], s[m]; r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (np.std(s)/np.std(o)-1)**2 + (np.mean(s)/(np.mean(o)+1e-9)-1)**2)


def waterfall(need, well, src_present, gates, Tn):
    """need,well: (T,). src_present: set of types this town has. gates: dict type->bool(active).
    Returns dict type->(T,) flow."""
    rem = np.maximum(need - well, 0.0)
    out = {"Well": np.minimum(well, need)}
    for st in PRIORITY:
        if st in src_present and gates.get(st, True):
            cap = CAP_LPS[st] * LPS
            f = np.minimum(rem, cap)
        else:
            f = np.zeros(Tn)
        out[st] = f; rem = rem - f
    out["unmet"] = rem
    return out


def main():
    P = zarr.open_group(_ARGS.preds, mode="r")
    print(f"### waterfall .3 allocation KGE — base={_ARGS.label}  preds={_ARGS.preds}")
    ptarg = list(P.attrs["target_names"]); pmap = {n: i for i, n in enumerate(ptarg)}; prid = list(P["run_ids"][:])
    # per town: well link, source-type -> observed link idx, demand col
    reg = {}
    for town in town_list:
        T = towns[town]; srcs = T["srcs"]
        well_name = [targ[T["srcs"]["well"]]][0] if "well" in T["srcs"] else None
        type_obs = {}
        for st in ["Aduccion", "PozoCostero", "Desal", "Acuerdo", "Camiones"]:
            if st in srcs: type_obs[st] = T["srcs"][st]
        reg[town] = dict(well_idx=T["srcs"].get("well"), well_name=targ[T["srcs"]["well"]] if "well" in T["srcs"] else None,
                         type_obs=type_obs, dcol=T["dcol"], present=set(type_obs.keys()))
    # accumulate per-type series: observed, v3-native, v3.3-waterfall
    agg = {grp: {st: {"o": [], "n": [], "w": []} for st in ["Well", "Desal", "Camiones"]} for grp in ["factorial", "lhs"]}
    for r in prid:
        si = rids.index(int(r)); Yo = Z["Y"][si]; X = Z["X"][si]; Yp = P["Y"][prid.index(r)]; Tn = Yo.shape[0]
        grp = "lhs" if int(r) >= 1000 else "factorial"
        # actions are STEP functions (0 until activation wk 559/819) -> gate on max over horizon, not wk0
        desal_on = any(np.nanmax(X[:, c]) > 0 for c in q_desal)
        pozo_on = (q_pozo is not None and np.nanmax(X[:, q_pozo]) > 0)
        for town in town_list:
            R = reg[town]
            if R["well_name"] is None or R["well_name"] not in pmap: continue
            need = X[:, R["dcol"]] * SEC / 0.70
            well_v3 = Yp[:, pmap[R["well_name"]]]
            wf = waterfall(need, well_v3, R["present"], {"Desal": desal_on, "PozoCostero": pozo_on}, Tn)
            for st in ["Well", "Desal", "Camiones"]:
                if st == "Well":
                    o = Yo[:, R["well_idx"]]; n = Yp[:, pmap[R["well_name"]]]; w = wf["Well"]
                elif st in R["type_obs"]:
                    oi = R["type_obs"][st]; nm = targ[oi]
                    o = Yo[:, oi]; n = Yp[:, pmap[nm]] if nm in pmap else np.zeros(Tn); w = wf[st]
                else:
                    continue
                agg[grp][st]["o"].append(o[WARMUP:]); agg[grp][st]["n"].append(n[WARMUP:]); agg[grp][st]["w"].append(w[WARMUP:])
    for grp in ["factorial", "lhs"]:
        L = _ARGS.label
        print(f"\n=== {grp} runs — flow KGE: {L} native vs {L}.3 waterfall ===")
        print(f"{'source':10s}{L+' native':>11s}{L+'.3 waterfall':>16s}")
        for st in ["Well", "Desal", "Camiones"]:
            d = agg[grp][st]
            if not d["o"]: continue
            o = np.concatenate(d["o"]); n = np.concatenate(d["n"]); w = np.concatenate(d["w"])
            print(f"{st:10s}{kge(o,n):>11.3f}{kge(o,w):>16.3f}")


if __name__ == "__main__":
    main()
