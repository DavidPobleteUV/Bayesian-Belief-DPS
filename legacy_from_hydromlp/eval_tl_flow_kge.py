# -*- coding: utf-8 -*-
"""
eval_tl_flow_kge.py — KGE of transmission-link flows INTO towns, by source type.
Water-balance view (not cost). Compares v3 native transmission vs v5.1 allocation
head (both deployed on v3's predicted GW), weekly, per source type.
"""
from __future__ import annotations
import sys
import numpy as np, zarr, pandas as pd, torch
sys.path.insert(0, "src")
from rdm_mlp.allocation.allocation_torch import LearnedAllocationHead
from train_v5_allocation import build_registry, build_well_caps, SEC, WARMUP

Z = zarr.open_group("data/weap_weekly.zarr", mode="r")
targ = list(Z.attrs["target_names"]); feat = list(Z.attrs["feature_names"]); rids = list(Z["run_ids"][:])
fi = {n: i for i, n in enumerate(feat)}
cost = pd.read_csv("../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv")
cost["withdrawal_node"] = cost["withdrawal_node"].astype(str).str.strip()
ncost = {r.withdrawal_node: r.source_type for r in cost.itertuples()}
def stype(name):
    src = name.split("__", 1)[1].rsplit("_to_", 1)[0].replace("Transmission_Link_from_", "")
    if ("APR_" in src or "APU_" in src) and "_Fict_" in src: return "Well"
    if src.startswith("Withdrawal_Node_"): return ncost.get(src.replace("Withdrawal_Node_", ""), "?")
    if src.startswith("DemAGRO_SHAC"): return "Acuerdo"
    return "?"

def kge(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float); m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 30 or np.std(o[m]) < 1e-9: return np.nan
    o, s = o[m], s[m]; r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (np.std(s)/np.std(o)-1)**2 + (np.mean(s)/(np.mean(o)+1e-9)-1)**2)

well_caps = build_well_caps("data/Q_wells.xlsx")
towns, ti, _ = build_registry(Z, well_caps, cost)
town_list = sorted(towns)
ck = torch.load("runs/iter05_v5p1/alloc_head_best.pt", weights_only=False)
mu, sd, Din = ck["mu"], ck["sd"], ck["Din"]; alloc_cols = ck["alloc_cols"]; src_types = ck["src_types"]; tourist = ck["tourist"]
head = LearnedAllocationHead(Din, ck["src_col"], ck["src_mask"], ck["src_cap"], len(alloc_cols), hidden=ck["hidden"])
head.load_state_dict(ck["state_dict"]); head.eval()
n_towns, K = len(town_list), len(src_types)
act_cols = [fi[n] for n in ["q_desalacion_costera","q_desalacion_completa","q_prorrateo_shac","q_prorrateo_cuenca","q_nuevo_pozo_a_5km"] if n in fi]
gate = ck.get("gate"); desal_qi = [fi[n] for n in gate["desal_feats"] if n in fi]
pozo_qi = fi.get(gate["pozo_feat"]) if gate["pozo_feat"] else None; SD, SP = gate["slot_desal"], gate["slot_pozo"]

P = zarr.open_group("results/preds_v3_j6spread.zarr", mode="r"); ptarg = list(P.attrs["target_names"]); pmap = {n: i for i, n in enumerate(ptarg)}
prid = list(P["run_ids"][:])
def to_pred(idxs): return [pmap[targ[i]] for i in idxs if targ[i] in pmap]
gwmap = {t: {"d": to_pred(towns[t]["depth"]), "s": to_pred(towns[t]["sal"]),
             "st": pmap.get(targ[towns[t]["stor"]]) if towns[t]["stor"] is not None else None} for t in town_list}
alloc_type = [stype(targ[c]) for c in alloc_cols]

# gather flows (nrun, T, n_alloc): observed, v3 native, v5.1 head
OBS, V3, V5 = [], [], []
for r in prid:
    si = rids.index(int(r)); Xo = Z["X"][si]; Yp = P["Y"][prid.index(r)]; Tn = Yp.shape[0]
    OBS.append(Z["Y"][si][:, alloc_cols])
    V3.append(np.stack([Yp[:, pmap[targ[c]]] if targ[c] in pmap else np.zeros(Tn) for c in alloc_cols], 1))
    drv = np.zeros((Tn, Din), np.float32); k = 0
    for t in town_list:
        g = gwmap[t]
        depth = Yp[:, g["d"]].mean(1) if g["d"] else np.zeros(Tn)
        sal = Yp[:, g["s"]].mean(1) if (towns[t]["coastal"] and g["s"]) else np.zeros(Tn)
        sl = np.concatenate([[sal[0]], sal[:-1]]); stor = Yp[:, g["st"]] if g["st"] is not None else np.zeros(Tn)
        drv[:, k] = depth; drv[:, k+1] = sal; drv[:, k+2] = sl; drv[:, k+3] = stor; k += 4
    drv[:, k:k+len(act_cols)] = Xo[:, act_cols]; drv[:, n_towns*4+5] = np.arange(Tn)/Tn
    drv = (drv-mu)/sd
    dem = np.stack([Xo[:, towns[t]["dcol"]] * SEC * tourist[t] for t in town_list], 1).astype(np.float32)
    gm = np.ones((n_towns, K), bool); x0 = Xo[0]; gm[:, SD] = any(x0[c] > 0 for c in desal_qi)
    if pozo_qi is not None: gm[:, SP] = x0[pozo_qi] > 0
    with torch.no_grad():
        rec, _, _ = head(torch.tensor(drv)[None], torch.tensor(dem)[None], dyn_mask=torch.tensor(gm)[None, None])
    V5.append(rec[0].numpy())
OBS, V3, V5 = np.stack(OBS), np.stack(V3), np.stack(V5)

print(f"Per-source-type TL flow KGE (weekly, deployed, n={len(prid)} runs)")
print(f"{'source':12s}{'#links':>7s}{'obs share':>11s}{'v3 native':>11s}{'v5.1 head':>11s}")
for t in ["Well","Camiones","Desal","Aduccion","PozoCostero","Acuerdo"]:
    cols = [j for j, tt in enumerate(alloc_type) if tt == t]
    if not cols: continue
    o = OBS[:, WARMUP:, cols].sum(2).ravel(); v3 = V3[:, WARMUP:, cols].sum(2).ravel(); v5 = V5[:, WARMUP:, cols].sum(2).ravel()
    share = OBS[:, WARMUP:, cols].sum() / OBS[:, WARMUP:, :].sum()
    print(f"{t:12s}{len(cols):>7d}{share:>10.1%}{kge(o,v3):>11.3f}{kge(o,v5):>11.3f}")
# total supply into towns (water balance)
o = OBS[:, WARMUP:, :].sum(2).ravel(); v3 = V3[:, WARMUP:, :].sum(2).ravel(); v5 = V5[:, WARMUP:, :].sum(2).ravel()
print(f"{'TOTAL supply':12s}{len(alloc_cols):>7d}{1.0:>10.1%}{kge(o,v3):>11.3f}{kge(o,v5):>11.3f}")
