# -*- coding: utf-8 -*-
"""
eval_transmission_aggregate.py — score transmission the way J4 (cost) and J2 (unmet)
consume it: aggregate scalars per scenario, predicted vs observed, ranking + bias.
Compares v3 (native cascade transmission) vs v4 on the same runs, to decide whether
v3.1 needs the v5.1 allocation head for cost/unmet.
"""
from __future__ import annotations
import numpy as np, zarr, pandas as pd, re
SEC = 604800.0
Z = zarr.open_group("data/weap_weekly.zarr", mode="r")
targ = list(Z.attrs["target_names"]); rids = list(Z["run_ids"][:]); ti = {n: i for i, n in enumerate(targ)}

# AP unmet targets (J2)
ap_unmet = [i for i, n in enumerate(targ) if "UnmetDemand" in n and re.search(r"APR|APU|AP_", n)]
if not ap_unmet:  # fallback: any AP-zone unmet
    ap_unmet = [i for i, n in enumerate(targ) if "UnmetDemand" in n and "Q0" in n and "Agricola" not in n]
print(f"AP unmet targets: {len(ap_unmet)}  e.g. {[targ[i] for i in ap_unmet[:3]]}")

# link -> unit cost (CLP/m3) for J4 (OPEX proxy: ranking driven by truck/desal usage)
cost = pd.read_csv("../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv")
cost["withdrawal_node"] = cost["withdrawal_node"].astype(str).str.strip()
node_cost = {r.withdrawal_node: float(r.unit_cost_clp_m3) for r in cost.itertuples()}
UNIT = {}
for n in targ:
    if "AP_TransmissionLinks" not in n:
        continue
    src = n.split("__", 1)[1].rsplit("_to_", 1)[0].replace("Transmission_Link_from_", "")
    if ("APR_" in src or "APU_" in src) and "_Fict_" in src:
        UNIT[ti[n]] = 200.0                        # own well: cheap pumping proxy
    elif src.startswith("Withdrawal_Node_"):
        UNIT[ti[n]] = node_cost.get(src.replace("Withdrawal_Node_", ""), 1500.0)
    elif src.startswith("DemAGRO_SHAC"):
        UNIT[ti[n]] = 3500.0                        # acuerdo
link_idx = list(UNIT.keys()); link_cost = np.array([UNIT[i] for i in link_idx])

WARMUP = 104
def metrics(Y):
    j2 = float(np.maximum(Y[WARMUP:][:, ap_unmet] * SEC, 0).sum())          # m3 unmet
    j4 = float((np.maximum(Y[WARMUP:][:, link_idx], 0) * link_cost).sum())   # CLP opex
    return j2, j4

def corr(a, b):
    a, b = np.asarray(a), np.asarray(b); m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 3 and np.std(a[m]) > 0 else np.nan

def score(label, pred_zarr):
    P = zarr.open_group(pred_zarr, mode="r"); ptarg = list(P.attrs["target_names"])
    pmap = {n: i for i, n in enumerate(ptarg)}
    # remap our target indices (full 767) to preds (filtered) by name
    pa_un = [pmap[targ[i]] for i in ap_unmet if targ[i] in pmap]
    pa_lk = [pmap[targ[i]] for i in link_idx if targ[i] in pmap]
    pa_lc = np.array([UNIT[i] for i in link_idx if targ[i] in pmap])
    prid = list(P["run_ids"][:])
    oJ2 = []; oJ4 = []; pJ2 = []; pJ4 = []
    for r in prid:
        si = rids.index(int(r)); Yo = Z["Y"][si]; Yp = P["Y"][prid.index(r)]
        o2, o4 = metrics(Yo); oJ2.append(o2); oJ4.append(o4)
        pJ2.append(float(np.maximum(Yp[WARMUP:][:, pa_un] * SEC, 0).sum()))
        pJ4.append(float((np.maximum(Yp[WARMUP:][:, pa_lk], 0) * pa_lc).sum()))
    oJ2, oJ4, pJ2, pJ4 = map(np.array, (oJ2, oJ4, pJ2, pJ4))
    def b(o, p): return (np.mean(p) - np.mean(o)) / (np.mean(o) + 1e-9)
    print(f"\n=== {label}  (n={len(prid)} runs) ===")
    print(f"  J2 unmet:  ranking corr={corr(pJ2,oJ2):.3f}  rel.bias={b(oJ2,pJ2):+.1%}   obs mean={oJ2.mean():.2e}  pred mean={pJ2.mean():.2e}")
    print(f"  J4 cost :  ranking corr={corr(pJ4,oJ4):.3f}  rel.bias={b(oJ4,pJ4):+.1%}   obs mean={oJ4.mean():.2e}  pred mean={pJ4.mean():.2e}")

score("v3 native transmission (cascade)", "results/preds_v3_j6spread.zarr")
score("v4 transmission", "results/preds_v4_j6spread.zarr")
