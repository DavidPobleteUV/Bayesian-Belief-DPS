# -*- coding: utf-8 -*-
"""
eval_v51_j4.py — does the v5.1 allocation head give a more faithful J4 (cost)?
Runs the gated allocation head on v3's predicted GW (J6-spread runs), computes J4 by
source type, and compares the truck/desal bias vs v3 native transmission.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np, zarr, pandas as pd, torch
sys.path.insert(0, "src")
from rdm_mlp.allocation.allocation_torch import LearnedAllocationHead
from train_v5_allocation import build_registry, build_well_caps, SEC, WARMUP

Z = zarr.open_group("data/weap_weekly.zarr", mode="r")
targ = list(Z.attrs["target_names"]); feat = list(Z.attrs["feature_names"]); rids = list(Z["run_ids"][:])
fi = {n: i for i, n in enumerate(feat)}
cost = pd.read_csv("../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv")
cost["withdrawal_node"] = cost["withdrawal_node"].astype(str).str.strip()
ncost = {r.withdrawal_node: (r.source_type, float(r.unit_cost_clp_m3)) for r in cost.itertuples()}
def classify(name):
    src = name.split("__", 1)[1].rsplit("_to_", 1)[0].replace("Transmission_Link_from_", "")
    if ("APR_" in src or "APU_" in src) and "_Fict_" in src: return ("Well", 200.0)
    if src.startswith("Withdrawal_Node_"): return ncost.get(src.replace("Withdrawal_Node_", ""), ("?", 1500.0))
    if src.startswith("DemAGRO_SHAC"): return ("Acuerdo", 3500.0)
    return ("?", 0.0)

well_caps = build_well_caps("data/Q_wells.xlsx")
towns, ti, fi2 = build_registry(Z, well_caps, cost)
town_list = sorted(towns)
ck = torch.load("runs/iter05_v5p1/alloc_head_best.pt", weights_only=False)
mu, sd, Din = ck["mu"], ck["sd"], ck["Din"]; alloc_cols = ck["alloc_cols"]; src_types = ck["src_types"]; tourist = ck["tourist"]
head = LearnedAllocationHead(Din, ck["src_col"], ck["src_mask"], ck["src_cap"], len(alloc_cols), hidden=ck["hidden"])
head.load_state_dict(ck["state_dict"]); head.eval()
n_towns, K = len(town_list), len(src_types)
act_cols = [fi[n] for n in ["q_desalacion_costera","q_desalacion_completa","q_prorrateo_shac","q_prorrateo_cuenca","q_nuevo_pozo_a_5km"] if n in fi]
gate = ck.get("gate")
desal_qi = [fi[n] for n in gate["desal_feats"] if n in fi]; pozo_qi = fi.get(gate["pozo_feat"]) if gate["pozo_feat"] else None
SD, SP = gate["slot_desal"], gate["slot_pozo"]

P = zarr.open_group("results/preds_v3_j6spread.zarr", mode="r")
ptarg = list(P.attrs["target_names"]); pmap = {n: i for i, n in enumerate(ptarg)}
prid = list(P["run_ids"][:])
def to_pred(idxs): return [pmap[targ[i]] for i in idxs if targ[i] in pmap]
gwmap = {t: {"d": to_pred(towns[t]["depth"]), "s": to_pred(towns[t]["sal"]),
             "st": pmap.get(targ[towns[t]["stor"]]) if towns[t]["stor"] is not None else None} for t in town_list}

# alloc column -> (type, unit_cost)
alloc_meta = [classify(targ[c]) for c in alloc_cols]

def J4_from_flows(flow, idx_runs):
    """flow: (nrun,T,n_alloc) raw m3/wk. Return dict type->cost over horizon."""
    by = {t: 0.0 for t in ["Well","Aduccion","Desal","PozoCostero","Camiones","Acuerdo"]}
    for j, (t, c) in enumerate(alloc_meta):
        if t in by: by[t] += float(np.maximum(flow[:, WARMUP:, j], 0).sum() * c)
    return by

# observed J4 (from observed flows at alloc_cols)
obs_flow = np.stack([Z["Y"][rids.index(int(r))][:, alloc_cols] for r in prid])
oc = J4_from_flows(obs_flow, prid)

# v5.1 predicted flows (head on v3 predicted GW)
recon_all = []
for r in prid:
    si = rids.index(int(r)); Xo = Z["X"][si]; Yp = P["Y"][prid.index(r)]
    Tn = Yp.shape[0]; drv = np.zeros((Tn, Din), dtype=np.float32); k = 0
    for t in town_list:
        g = gwmap[t]
        depth = Yp[:, g["d"]].mean(1) if g["d"] else np.zeros(Tn)
        sal = Yp[:, g["s"]].mean(1) if (towns[t]["coastal"] and g["s"]) else np.zeros(Tn)
        sal_lag = np.concatenate([[sal[0]], sal[:-1]])
        stor = Yp[:, g["st"]] if g["st"] is not None else np.zeros(Tn)
        drv[:, k] = depth; drv[:, k+1] = sal; drv[:, k+2] = sal_lag; drv[:, k+3] = stor; k += 4
    drv[:, k:k+len(act_cols)] = Xo[:, act_cols]; drv[:, n_towns*4+5] = np.arange(Tn)/Tn
    drv = (drv - mu) / sd
    dem = np.stack([Xo[:, towns[t]["dcol"]] * SEC * tourist[t] for t in town_list], 1).astype(np.float32)
    gmask = np.ones((n_towns, K), bool); x0 = Xo[0]
    gmask[:, SD] = any(x0[c] > 0 for c in desal_qi)
    if pozo_qi is not None: gmask[:, SP] = x0[pozo_qi] > 0
    with torch.no_grad():
        rec, _, _ = head(torch.tensor(drv)[None], torch.tensor(dem)[None], dyn_mask=torch.tensor(gmask)[None, None])
    recon_all.append(rec[0].numpy())
pc = J4_from_flows(np.stack(recon_all), prid)

# v3 native (recompute from v3 preds at alloc_cols)
v3_flow = np.stack([P["Y"][prid.index(r)][:, [pmap[targ[c]] for c in alloc_cols]] for r in prid])
v3c = J4_from_flows(v3_flow, prid)

tot_o = sum(oc.values())
print(f"{'source':12s}{'obs':>10s}{'v3 native':>12s}{'v5.1 head':>12s}{'%J4':>7s}{'v3 bias':>9s}{'v5.1 bias':>10s}")
for t in ["Camiones","Desal","Well","Aduccion","PozoCostero","Acuerdo"]:
    o, v3, v5 = oc[t], v3c[t], pc[t]
    print(f"{t:12s}{o/1e9:>9.0f}B{v3/1e9:>11.0f}B{v5/1e9:>11.0f}B{100*o/tot_o:>6.0f}%"
          f"{(v3/max(o,1)-1)*100:>+8.0f}%{(v5/max(o,1)-1)*100:>+9.0f}%")
to,t3,t5 = tot_o, sum(v3c.values()), sum(pc.values())
print(f"{'TOTAL':12s}{to/1e9:>9.0f}B{t3/1e9:>11.0f}B{t5/1e9:>11.0f}B{100:>6.0f}%{(t3/to-1)*100:>+8.0f}%{(t5/to-1)*100:>+9.0f}%")
