# -*- coding: utf-8 -*-
"""
kge_variables_table.py — KGE (promedio y mediana sobre runs de test) por MLP base
para 5 variables de interes:
  GW storage          = suma SHAC_storage_Acuifero_Q01..Q09  (target directo)
  Avocado production  = suma AGR_AnnualCropProduction Palto    (target directo)
  Unmet potable       = suma AP_UnmetDemand (APR+APU)          (target directo)
  Supply cost         = Σ links_fallback flow×unit_cost        (derivada de asignacion)
  Truck water share   = Camiones / (well + fallback)           (derivada de asignacion)

KGE por run sobre la ventana de decision (WARMUP:fin); luego promedio y mediana
nan-robustos sobre los runs. Una columna por MLP base (--preds).
  python kge_variables_table.py --preds results/preds_v3clean.zarr --label v3
"""
from __future__ import annotations
import argparse, sys
import numpy as np, zarr, pandas as pd
sys.path.insert(0, ".")
from train_v5_allocation import build_registry, build_well_caps, SEC, WARMUP

LPS = 604.8
UNIT_COST = {"Aduccion": 500.0, "PozoCostero": 1200.0, "Desal": 1500.0,
             "Camiones": 8000.0, "Acuerdo": 3500.0}
FALLBACK = ["Aduccion", "PozoCostero", "Desal", "Camiones", "Acuerdo"]
CAP_LPS = {"Aduccion": 2.0, "PozoCostero": 120.0, "Desal": 1e9, "Camiones": 1e9}
WF_PRIORITY = ["Aduccion", "PozoCostero", "Desal", "Camiones"]   # Acuerdo excluido en .3

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--label", default="?")
ap.add_argument("--waterfall", action="store_true",
                help=".3: deriva cost/truck con la cascada well-anclada (desal/camiones = deficit)")
args = ap.parse_args()

Z = zarr.open_group("data/weap_weekly.zarr", mode="r")
targ = list(Z.attrs["target_names"]); rids = list(Z["run_ids"][:])
ti = {n: i for i, n in enumerate(targ)}
cost = pd.read_csv("../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv")
cost["withdrawal_node"] = cost["withdrawal_node"].astype(str).str.strip()
well_caps = build_well_caps("data/Q_wells.xlsx")
towns, _, _ = build_registry(Z, well_caps, cost)

feat = list(Z.attrs["feature_names"]); fi = {n: i for i, n in enumerate(feat)}
q_desal = [fi[n] for n in ["q_desalacion_costera", "q_desalacion_completa"] if n in fi]
q_pozo = fi.get("q_nuevo_pozo_a_5km")

P = zarr.open_group(args.preds, mode="r")
ptarg = list(P.attrs["target_names"]); pmap = {n: i for i, n in enumerate(ptarg)}
prid = [int(x) for x in P["run_ids"][:]]

# target index groups (en el espacio FULL de Z)
stor_cols = [ti[f"SHAC_storage_Acuifero_Q0{i}_MF_m3"] for i in range(1, 10)
             if f"SHAC_storage_Acuifero_Q0{i}_MF_m3" in ti]
avoc_cols = [i for i, n in enumerate(targ) if "AGR_AnnualCropProduction" in n and "Palto" in n]
unmet_cols = [i for i, n in enumerate(targ) if "AP_UnmetDemand" in n]

def kge(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 30 or np.std(o[m]) < 1e-9:
        return np.nan
    o, s = o[m], s[m]; r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (np.std(s)/np.std(o)-1)**2 + (np.mean(s)/(np.mean(o)+1e-9)-1)**2)

def series_full(Y, cols):
    return Y[WARMUP:, cols].sum(1) if cols else np.zeros(Y.shape[0]-WARMUP)

# para pred (P) hay que mapear los nombres -> indices de P
def pcols(cols):
    return [pmap[targ[c]] for c in cols if targ[c] in pmap]

per_run = {k: [] for k in ["GW storage", "Avocado production", "Unmet potable",
                            "Supply cost", "Truck water share"]}

for r in prid:
    si = rids.index(int(r)); Yo = Z["Y"][si]; Yp = P["Y"][prid.index(r)]; X = Z["X"][si]
    # --- targets directos (no cambian con waterfall) ---
    per_run["GW storage"].append(kge(series_full(Yo, stor_cols), series_full(Yp, pcols(stor_cols))))
    per_run["Avocado production"].append(kge(series_full(Yo, avoc_cols), series_full(Yp, pcols(avoc_cols))))
    per_run["Unmet potable"].append(kge(series_full(Yo, unmet_cols), series_full(Yp, pcols(unmet_cols))))
    # gates de accion (step: activo si max>0 en el horizonte)
    desal_on = any(np.nanmax(X[:, c]) > 0 for c in q_desal)
    pozo_on = (q_pozo is not None and np.nanmax(X[:, q_pozo]) > 0)
    gates = {"Desal": desal_on, "PozoCostero": pozo_on}
    # --- derivadas de asignacion (por town, sumando) ---
    o_cost = np.zeros(Yo.shape[0]-WARMUP); s_cost = np.zeros_like(o_cost)
    o_truck = np.zeros_like(o_cost); s_truck = np.zeros_like(o_cost)
    o_tot = np.zeros_like(o_cost); s_tot = np.zeros_like(o_cost)
    for town, T in towns.items():
        srcs = T["srcs"]
        # well nativo (parte del total de suministro)
        well_pred = np.zeros_like(o_cost)
        if "well" in srcs:
            o_tot += np.maximum(Yo[WARMUP:, srcs["well"]], 0.0)
            wn = targ[srcs["well"]]
            well_pred = np.maximum(Yp[WARMUP:, pmap[wn]], 0.0) if wn in pmap else np.zeros_like(o_cost)
            s_tot += well_pred
        # cascada waterfall (lado pred): deficit por prioridad de precio
        wf = {}
        if args.waterfall:
            need = np.maximum(X[WARMUP:, T["dcol"]], 0.0) * SEC / 0.70
            rem = np.maximum(need - well_pred, 0.0)
            for st in WF_PRIORITY:
                if st in srcs and gates.get(st, True):
                    f = np.minimum(rem, CAP_LPS[st] * LPS)
                else:
                    f = np.zeros_like(rem)
                wf[st] = f; rem = rem - f
        for st in FALLBACK:
            if st not in srcs:
                continue
            uc = UNIT_COST[st]; nm = targ[srcs[st]]
            o = np.maximum(Yo[WARMUP:, srcs[st]], 0.0)
            if args.waterfall:
                s = wf.get(st, np.zeros_like(o))      # Acuerdo -> 0 en .3
            else:
                s = np.maximum(Yp[WARMUP:, pmap[nm]], 0.0) if nm in pmap else np.zeros_like(o)
            o_cost += o * uc; s_cost += s * uc
            o_tot += o; s_tot += s
            if st == "Camiones":
                o_truck += o; s_truck += s
    per_run["Supply cost"].append(kge(o_cost, s_cost))
    # truck share = camiones / total (evita div0)
    o_sh = o_truck / np.where(o_tot > 1e-9, o_tot, np.nan)
    s_sh = s_truck / np.where(s_tot > 1e-9, s_tot, np.nan)
    per_run["Truck water share"].append(kge(o_sh, s_sh))

print(f"\n===== KGE por variable — MLP {args.label}  ({len(prid)} runs test) =====")
print(f"{'variable':22s}{'KGE promedio':>14s}{'KGE mediana':>14s}{'n_runs_valid':>14s}")
for k in ["GW storage", "Avocado production", "Unmet potable", "Supply cost", "Truck water share"]:
    a = np.array(per_run[k], float); v = a[np.isfinite(a)]
    print(f"{k:22s}{np.mean(v):>14.3f}{np.median(v):>14.3f}{len(v):>14d}")
