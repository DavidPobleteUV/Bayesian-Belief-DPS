# -*- coding: utf-8 -*-
"""
eval_v5_deployed.py — DEPLOYED-mode evaluation of the v5 allocation head.

Trains/scored teacher-forced (observed GW drivers) the head reached transm. KGE
0.67-0.81. In production there is no WEAP: the head must run on v4's *predicted* GW
state. This script measures that gap:

  observed GW  -> head -> KGE   (teacher-forced, the ceiling)
  v4-predicted GW -> head -> KGE   (deployed, what DPS will actually use)

Both compared against the SAME observed per-source flows.

  python eval_v5_deployed.py --head runs/iter04_v5/alloc_head_best.pt \
      --zarr data/weap_weekly.zarr --preds results/preds_v4_deployed.zarr
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, zarr, torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rdm_mlp.allocation.allocation_torch import LearnedAllocationHead
from train_v5_allocation import build_registry, build_well_caps, kge, WARMUP, SEC

COASTAL = {"ElEsfuerzo", "Quilimari", "Pichidangui"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="runs/iter04_v5/alloc_head_best.pt")
    ap.add_argument("--zarr", default="data/weap_weekly.zarr")
    ap.add_argument("--preds", default="results/preds_v4_deployed.zarr")
    ap.add_argument("--cost_csv", default="../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv")
    ap.add_argument("--q_wells", default="data/Q_wells.xlsx")
    args = ap.parse_args()

    Z = zarr.open_group(args.zarr, mode="r")
    P = zarr.open_group(args.preds, mode="r")
    targ = list(Z.attrs["target_names"]); feat = list(Z.attrs["feature_names"])
    ptarg = list(P.attrs["target_names"])
    pname2idx = {n: i for i, n in enumerate(ptarg)}   # preds Y is the filtered subset; align by NAME
    rids = list(Z["run_ids"][:]); prids = list(P["run_ids"][:])

    well_caps = build_well_caps(args.q_wells)
    cost = pd.read_csv(args.cost_csv)
    towns, ti, fi = build_registry(Z, well_caps, cost)
    town_list = sorted(towns)

    ck = torch.load(args.head, weights_only=False)
    mu, sd, Din = ck["mu"], ck["sd"], ck["Din"]
    alloc_cols = ck["alloc_cols"]; src_types = ck["src_types"]; tourist = ck["tourist"]
    head = LearnedAllocationHead(Din, ck["src_col"], ck["src_mask"], ck["src_cap"],
                                 len(alloc_cols), hidden=ck["hidden"])
    head.load_state_dict(ck["state_dict"]); head.eval()

    n_towns = len(town_list); K = len(src_types)
    act_cols = [fi[n] for n in ["q_desalacion_costera", "q_desalacion_completa", "q_prorrateo_shac",
                                "q_prorrateo_cuenca", "q_nuevo_pozo_a_5km"] if n in fi]
    # v5.1 action gate (rebuild from ckpt metadata, if present)
    gate = ck.get("gate")
    if gate:
        desal_qi = [fi[n] for n in gate["desal_feats"] if n in fi]
        pozo_qi = fi.get(gate["pozo_feat"]) if gate["pozo_feat"] else None
        SLOT_DESAL, SLOT_POZO = gate["slot_desal"], gate["slot_pozo"]

    def run_gate(X):
        g = np.ones((n_towns, K), dtype=bool)
        if gate:
            x0 = X[0]
            g[:, SLOT_DESAL] = any(x0[c] > 0 for c in desal_qi)
            if pozo_qi is not None:
                g[:, SLOT_POZO] = x0[pozo_qi] > 0
        return g

    # per-town GW-driver column indices in BOTH spaces: Z (767, observed) and P (733, predicted)
    def to_pred(idx_list):
        return [pname2idx[targ[i]] for i in idx_list if targ[i] in pname2idx]
    gwmap = {}
    for town in town_list:
        Tt = towns[town]
        sp = pname2idx.get(targ[Tt["stor"]]) if Tt["stor"] is not None else None
        gwmap[town] = {
            False: (Tt["depth"], Tt["sal"], Tt["stor"]),                 # observed (Z)
            True:  (to_pred(Tt["depth"]), to_pred(Tt["sal"]), sp),       # predicted (P)
        }

    def drivers_for(gwY, X, use_pred):
        """Build the driver matrix (T,Din) — GW state from gwY (observed Z OR predicted P)."""
        Tn = gwY.shape[0]; drv = np.zeros((Tn, Din), dtype=np.float32); k = 0
        for town in town_list:
            Tt = towns[town]; di, si, sti = gwmap[town][use_pred]
            depth = gwY[:, di].mean(1) if di else np.zeros(Tn)
            sal = gwY[:, si].mean(1) if (Tt["coastal"] and si) else np.zeros(Tn)
            sal_lag = np.concatenate([[sal[0]], sal[:-1]])
            stor = gwY[:, sti] if sti is not None else np.zeros(Tn)
            drv[:, k] = depth; drv[:, k+1] = sal; drv[:, k+2] = sal_lag; drv[:, k+3] = stor; k += 4
        if act_cols:
            a = X[:, act_cols]; drv[:, k:k+a.shape[1]] = a
        drv[:, n_towns*4+5] = np.arange(Tn) / Tn
        return drv

    def run_mode(use_pred):
        recon_all, obs_all = [], []
        for rid in prids:
            si = rids.index(rid); pi = prids.index(rid)
            Xo = Z["X"][si]; Yo = Z["Y"][si]
            gwY = P["Y"][pi] if use_pred else Yo            # predicted vs observed GW state
            drv = drivers_for(gwY, Xo, use_pred)
            drv = (drv - mu) / sd
            dem = np.stack([Xo[:, towns[t]["dcol"]] * SEC * tourist[t] for t in town_list], 1).astype(np.float32)
            g = torch.tensor(run_gate(Xo))[None, None]      # (1,1,n_towns,K) action gate
            with torch.no_grad():
                recon, _, _ = head(torch.tensor(drv)[None], torch.tensor(dem)[None], dyn_mask=g)
            recon_all.append(recon[0, WARMUP:].numpy())
            obs_all.append(Yo[WARMUP:][:, alloc_cols])
        return np.concatenate(recon_all), np.concatenate(obs_all)

    print(f"[deployed-eval] runs={prids}  towns={town_list}\n")
    results = {}
    for mode, use_pred in [("teacher (observed GW)", False), ("DEPLOYED (v4-predicted GW)", True)]:
        recon, obs = run_mode(use_pred)
        # per source-type KGE (aggregate columns of each type)
        by = {st: [] for st in src_types}
        j = 0
        for town in town_list:
            for st in src_types:
                if st in towns[town]["srcs"]:
                    by[st].append(kge(obs[:, j], recon[:, j])); j += 1
        results[mode] = {st: (np.nanmedian(v) if len(v) and np.any(np.isfinite(v)) else np.nan)
                         for st, v in by.items()}
        results[mode]["MEDIAN(all)"] = np.nanmedian([kge(obs[:, c], recon[:, c]) for c in range(obs.shape[1])])

    # report
    cols = ["well", "Desal", "Camiones", "Aduccion", "PozoCostero", "Acuerdo", "MEDIAN(all)"]
    print(f"{'mode':28s}" + "".join(f"{c:>13s}" for c in cols))
    for mode in results:
        row = results[mode]
        print(f"{mode:28s}" + "".join(f"{row.get(c, float('nan')):13.3f}" for c in cols))
    tf, dp = results["teacher (observed GW)"]["MEDIAN(all)"], results["DEPLOYED (v4-predicted GW)"]["MEDIAN(all)"]
    print(f"\n  median transmission KGE: teacher={tf:.3f}  deployed={dp:.3f}  drop={tf-dp:.3f}")
    print("  (deployed = what the DPS will actually use for J2 unmet / J4 cost)")


if __name__ == "__main__":
    main()
