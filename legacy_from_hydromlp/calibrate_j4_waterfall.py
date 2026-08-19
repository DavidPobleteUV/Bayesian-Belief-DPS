# -*- coding: utf-8 -*-
"""
calibrate_j4_waterfall.py — re-derive the DPS J4 cost calibration factor for the
NATIVE and .3-WATERFALL variants of the clean models.

factor = E[true_cost] / E[predicted_cost]   (aggregate, over test runs)

  true_cost      = observed transmission flows  x unit_cost          (WEAP ground truth)
  native_cost    = model-predicted flows        x unit_cost          (what DPS costs, native)
  waterfall_cost = well-anchored cascade flows  x unit_cost          (what DPS costs, .3)

Fallback sources only (Aduccion/PozoCostero/Desal/Camiones/Acuerdo) — the volume-
dependent, fixed-unit-cost OPEX the surrogate mis-counts and the waterfall reallocates.
Wells (electrical, native in both) are excluded; they ~cancel in the ratio.

  python calibrate_j4_waterfall.py --preds results/preds_v3clean.zarr --label v3
"""
from __future__ import annotations
import argparse, sys
import numpy as np, zarr, pandas as pd
sys.path.insert(0, ".")
from train_v5_allocation import build_registry, build_well_caps, SEC, WARMUP

LPS = 604.8
CAP_LPS = {"Aduccion": 2.0, "PozoCostero": 120.0, "Desal": 1e9, "Camiones": 1e9}
PRIORITY = ["Aduccion", "PozoCostero", "Desal", "Camiones"]       # Acuerdo excluded in .3
UNIT_COST = {"Aduccion": 500.0, "PozoCostero": 1200.0, "Desal": 1500.0,
             "Camiones": 8000.0, "Acuerdo": 3500.0}
FALLBACK = ["Aduccion", "PozoCostero", "Desal", "Camiones", "Acuerdo"]

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--label", default="?")
args = ap.parse_args()

Z = zarr.open_group("data/weap_weekly.zarr", mode="r")
targ = list(Z.attrs["target_names"]); feat = list(Z.attrs["feature_names"]); rids = list(Z["run_ids"][:])
fi = {n: i for i, n in enumerate(feat)}
cost = pd.read_csv("../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv")
cost["withdrawal_node"] = cost["withdrawal_node"].astype(str).str.strip()
well_caps = build_well_caps("data/Q_wells.xlsx")
towns, ti, _ = build_registry(Z, well_caps, cost)

P = zarr.open_group(args.preds, mode="r")
ptarg = list(P.attrs["target_names"]); pmap = {n: i for i, n in enumerate(ptarg)}
prid = [int(x) for x in P["run_ids"][:]]

q_desal = [fi[n] for n in ["q_desalacion_costera", "q_desalacion_completa"] if n in fi]
q_pozo = fi.get("q_nuevo_pozo_a_5km")

# accumulators: source_type -> total CLP across all runs/towns
obs_c = {s: 0.0 for s in FALLBACK}
nat_c = {s: 0.0 for s in FALLBACK}
wf_c  = {s: 0.0 for s in FALLBACK}

for r in prid:
    si = rids.index(int(r)); Yo = Z["Y"][si]; X = Z["X"][si]; Yp = P["Y"][prid.index(r)]
    Tn = Yo.shape[0]
    desal_on = any(np.nanmax(X[:, c]) > 0 for c in q_desal)
    pozo_on = (q_pozo is not None and np.nanmax(X[:, q_pozo]) > 0)
    gates = {"Desal": desal_on, "PozoCostero": pozo_on}
    for town, T in towns.items():
        srcs = T["srcs"]
        if "well" not in srcs:
            continue
        well_name = targ[srcs["well"]]
        if well_name not in pmap:
            continue
        need = np.maximum(X[WARMUP:, T["dcol"]], 0.0) * SEC / 0.70
        well_pred = np.maximum(Yp[WARMUP:, pmap[well_name]], 0.0)
        rem = np.maximum(need - well_pred, 0.0)
        # waterfall fallback flows
        wf = {}
        for st in PRIORITY:
            if st in srcs and gates.get(st, True):
                f = np.minimum(rem, CAP_LPS[st] * LPS)
            else:
                f = np.zeros_like(rem)
            wf[st] = f; rem = rem - f
        # accumulate cost per fallback source
        for st in FALLBACK:
            if st not in srcs:
                continue
            uc = UNIT_COST[st]; oi = srcs[st]; nm = targ[oi]
            obs_c[st] += float(np.nansum(np.maximum(Yo[WARMUP:, oi], 0.0))) * uc
            nat_c[st] += float(np.nansum(np.maximum(Yp[WARMUP:, pmap[nm]], 0.0))) * uc if nm in pmap else 0.0
            wf_c[st]  += float(np.nansum(wf[st])) * uc if st in wf else 0.0   # Acuerdo -> 0 in wf

OBS = sum(obs_c.values()); NAT = sum(nat_c.values()); WF = sum(wf_c.values())
print(f"\n===== J4 calibration  base={args.label}  ({len(prid)} test runs) =====")
print(f"{'source':12s}{'OBS (truth)':>16s}{'NATIVE pred':>16s}{'.3 waterfall':>16s}")
for st in FALLBACK:
    print(f"{st:12s}{obs_c[st]:>16.3e}{nat_c[st]:>16.3e}{wf_c[st]:>16.3e}")
print(f"{'TOTAL':12s}{OBS:>16.3e}{NAT:>16.3e}{WF:>16.3e}")
print(f"\nfactor_native ({args.label})   = OBS/NATIVE = {OBS/NAT:6.3f}   (current config uses 1.22)")
print(f"factor_.3     ({args.label}.3) = OBS/WATERFALL = {OBS/WF:6.3f}   <-- new calibration for {args.label}.3")
