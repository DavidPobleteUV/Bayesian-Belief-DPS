# -*- coding: utf-8 -*-
"""
train_v5_allocation.py — Train the v5 LEARNED constrained allocation head (option B).

Standalone (does NOT touch v4): trains a small physical head
   f(GW drivers: depth-to-water, salinity, storage | demand | actions) -> per-source flows
in RAW space against the CLEAN observed AP transmission flows. This learns the true
(data-consistent) supply allocation, sidestepping the broken WF_PumpFactor export.

Gate metric = per-source transmission KGE on held-out runs (the corner v2/v3/v4 underfit).

  python train_v5_allocation.py --zarr data/weap_weekly.zarr \
     --q_wells data/Q_wells.xlsx \
     --cost_csv ../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv \
     --out runs/iter04_v5 --epochs 300 [--smoke]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, zarr, torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rdm_mlp.allocation.allocation_torch import LearnedAllocationHead, alloc_loss, LOSS_FACTOR

WARMUP = 104
SEC = 604800.0; LPS = 604.8
COASTAL = {"ElEsfuerzo", "Quilimari", "Pichidangui"}
# nominal source-capacity upper bounds (l/s) -> m³/wk (loose; head learns within)
CAP_LPS = {"Aduccion": 2.0, "PozoCostero": 120.0, "Desal": 300.0, "Camiones": 50.0}


def kge(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 10 or np.std(obs[m]) < 1e-9:
        return np.nan
    o, s = obs[m], sim[m]
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2 + (np.mean(s) / (np.mean(o) + 1e-9) - 1) ** 2)


def build_well_caps(p):
    """Load pre-extracted {fict_node: cap_lps} from data/well_caps.json (no openpyxl
    needed at train time). Falls back to the xlsx if the json is absent."""
    jp = Path(p).with_name("well_caps.json")
    if jp.exists():
        return {k: float(v) for k, v in json.load(open(jp)).items()}
    raw = pd.read_excel(p, header=None)
    hr = raw.index[raw.apply(lambda r: r.astype(str).str.contains("BranchID").any(), axis=1)][0]
    df = pd.read_excel(p, header=hr); df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Level 2": "node", "Variable": "var", "Expression": "expr"})
    cap = df[df["var"].astype(str).str.strip() == "Caudal asociado"].copy()
    cap["expr"] = pd.to_numeric(cap["expr"], errors="coerce")
    caps = {}
    for _, r in cap.iterrows():
        c = r["expr"]
        if np.isfinite(c) and c > 0:
            caps[str(r["node"])] = caps.get(str(r["node"]), 0.0) + float(c)
    return caps


def build_registry(Z, well_caps, cost):
    targ = list(Z.attrs["target_names"]); feat = list(Z.attrs["feature_names"])
    ti = {n: i for i, n in enumerate(targ)}; fi = {n: i for i, n in enumerate(feat)}
    links = [n for n in targ if "AP_TransmissionLinks" in n and "_to_" in n]
    dem_nodes = sorted({n.split("__", 1)[1].rsplit("_to_", 1)[1] for n in links})
    cost["withdrawal_node"] = cost["withdrawal_node"].astype(str).str.strip()
    towns = {}
    for dem in dem_nodes:
        town = re.search(r"_Dem_(\w+)", dem).group(1)
        if town == "ElManzanoL":
            continue
        shac = re.search(r"Q(\d+)", dem).group(1).zfill(2)
        dcol = fi.get("AP_WaterDemand__" + dem)
        my = [n for n in links if n.endswith("_to_" + dem)]
        srcs = {}; fict = None
        for n in my:
            src = n.split("__", 1)[1].rsplit("_to_", 1)[0].replace("Transmission_Link_from_", "")
            if ("APR_" in src or "APU_" in src) and "_Fict_" in src:
                srcs["well"] = ti[n]; fict = src
            elif src.startswith("Withdrawal_Node_"):
                node = src.replace("Withdrawal_Node_", "")
                row = cost[cost["withdrawal_node"] == node]
                if len(row):
                    srcs[row.iloc[0]["source_type"]] = ti[n]
            elif src.startswith("DemAGRO_SHAC"):
                srcs["Acuerdo"] = ti[n]
        if dcol is None or fict is None:
            continue
        well_cap_lps = float(well_caps.get(fict, 0.0))
        # gw driver cols
        depth = [ti[n] for n in targ if n.startswith(f"WF_DepthToWater_m__{fict}__")]
        sal = [ti[n] for n in targ if n.startswith(f"WF_SalinityFactor__{fict}__")]
        stor = ti.get(f"SHAC_storage_Acuifero_Q{shac}_MF_m3")
        towns[town] = dict(dem=dem, fict=fict, dcol=dcol, srcs=srcs, coastal=town in COASTAL,
                           depth=depth, sal=sal, stor=stor, well_cap_lps=well_cap_lps)
    return towns, ti, fi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/weap_weekly.zarr")
    ap.add_argument("--q_wells", default="data/Q_wells.xlsx")
    ap.add_argument("--cost_csv", default="../Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv")
    ap.add_argument("--out", default="runs/iter04_v5")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="few steps only, sanity check")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    Z = zarr.open_group(args.zarr, mode="r"); rids = Z["run_ids"][:]
    well_caps = build_well_caps(args.q_wells)
    cost = pd.read_csv(args.cost_csv)
    towns, ti, fi = build_registry(Z, well_caps, cost)
    town_list = sorted(towns)
    print(f"[v5] {len(town_list)} towns: {town_list}")

    # ── alloc-column order + per-town source slots + caps ──
    src_types = ["well", "Aduccion", "PozoCostero", "Desal", "Acuerdo", "Camiones"]
    alloc_cols = []          # zarr target idx per alloc output column
    src_col = []; src_mask = []; src_cap = []
    for town in town_list:
        T = towns[town]; row_idx = []; row_mask = []; row_cap = []
        for st in src_types:
            if st in T["srcs"]:
                row_idx.append(len(alloc_cols)); row_mask.append(True)
                alloc_cols.append(T["srcs"][st])
                if st == "well":
                    row_cap.append(max(T["well_cap_lps"], 1.0) * LPS)
                elif st == "Acuerdo":
                    row_cap.append(float("inf"))            # agreement: uncapped backup
                else:
                    row_cap.append(CAP_LPS[st] * LPS)
            else:
                row_idx.append(0); row_mask.append(False); row_cap.append(float("inf"))
        src_col.append(row_idx); src_mask.append(row_mask); src_cap.append(row_cap)
    n_alloc = len(alloc_cols); n_towns = len(town_list); K = len(src_types)
    src_col = torch.tensor(src_col); src_mask = torch.tensor(src_mask); src_cap = torch.tensor(src_cap)

    # ── tourist_mult calibration (observed AP supply / (demand/0.70)) ──
    tourist = {}
    ap_idx_per_town = {town: [towns[town]["srcs"][s] for s in src_types if s in towns[town]["srcs"] and s != "Acuerdo"]
                       for town in town_list}
    sample_runs = [int(r) for r in rids if r < 1000][:8]
    for town in town_list:
        rr = []
        for rid in sample_runs:
            s = int(np.where(rids == rid)[0][0]); Y = Z["Y"][s]; X = Z["X"][s]
            sup = Y[WARMUP:][:, ap_idx_per_town[town]].sum(1)
            need = X[WARMUP:, towns[town]["dcol"]] * SEC / LOSS_FACTOR
            m = need > 1
            if m.sum():
                rr.append(np.nanmedian(sup[m] / need[m]))
        tourist[town] = float(np.clip(np.nanmedian(rr) if rr else 1.0, 0.3, 3.0))
    print("[v5] tourist_mult:", {k: round(v, 2) for k, v in tourist.items()})

    # ── build per-run tensors: drivers, demand_raw, y_alloc_raw ──
    n_run = len(rids)
    Din = n_towns * 4 + 6 + 1   # per town: depth,sal,sal_lag,storage  + 5 actions + year + 1 bias-free
    # action feature columns (from X) — match canonical action q columns if present
    act_cols = [fi[n] for n in ["q_desalacion_costera", "q_desalacion_completa", "q_prorrateo_shac",
                                "q_prorrateo_cuenca", "q_nuevo_pozo_a_5km"] if n in fi]
    # v5.1 ACTION GATE — which q-columns switch on which source slot.
    gate_desal_feats = [n for n in ["q_desalacion_costera", "q_desalacion_completa"] if n in fi]
    gate_pozo_feat = "q_nuevo_pozo_a_5km" if "q_nuevo_pozo_a_5km" in fi else None
    SLOT_DESAL = src_types.index("Desal"); SLOT_POZO = src_types.index("PozoCostero")
    desal_qi = [fi[n] for n in gate_desal_feats]
    pozo_qi = fi.get(gate_pozo_feat) if gate_pozo_feat else None

    def run_gate(s):
        """Per-run source-active mask (n_towns, K): desal/pozo-costero OFF unless the
        action is built in this run (q>0). Other sources always available."""
        x0 = Z["X"][s][0]
        g = np.ones((n_towns, K), dtype=bool)
        g[:, SLOT_DESAL] = any(x0[c] > 0 for c in desal_qi)
        if pozo_qi is not None:
            g[:, SLOT_POZO] = x0[pozo_qi] > 0
        return g

    def run_tensors(s):
        Y = Z["Y"][s]; X = Z["X"][s]; Tn = Y.shape[0]
        drv = np.zeros((Tn, Din), dtype=np.float32); k = 0
        for town in town_list:
            Tt = towns[town]
            depth = Y[:, Tt["depth"]].mean(1) if Tt["depth"] else np.zeros(Tn)
            sal = Y[:, Tt["sal"]].mean(1) if (Tt["coastal"] and Tt["sal"]) else np.zeros(Tn)
            sal_lag = np.concatenate([[sal[0]], sal[:-1]])
            stor = Y[:, Tt["stor"]] if Tt["stor"] is not None else np.zeros(Tn)
            drv[:, k] = depth; drv[:, k + 1] = sal; drv[:, k + 2] = sal_lag; drv[:, k + 3] = stor
            k += 4
        if act_cols:
            a = X[:, act_cols]; drv[:, k:k + a.shape[1]] = a; k += a.shape[1]
        k = n_towns * 4 + 5
        drv[:, k] = (np.arange(Tn) / Tn)            # year fraction
        dem = np.stack([X[:, towns[t]["dcol"]] * SEC * tourist[t] for t in town_list], 1).astype(np.float32)
        yal = Y[:, alloc_cols].astype(np.float32)
        return drv, dem, yal

    # train/val split by run
    val_runs = set(int(r) for i, r in enumerate(rids) if i % 6 == 0)
    tr_idx = [i for i, r in enumerate(rids) if int(r) not in val_runs]
    va_idx = [i for i, r in enumerate(rids) if int(r) in val_runs]
    if args.smoke:
        tr_idx, va_idx = tr_idx[:6], va_idx[:3]
    print(f"[v5] runs train={len(tr_idx)} val={len(va_idx)}  Din={Din} n_alloc={n_alloc}")

    # materialize (small: ~768 × 1872 × ~60 floats)
    def stack(idxs):
        D, M, Yl = [], [], []
        for s in idxs:
            d, m, y = run_tensors(s); D.append(d); M.append(m); Yl.append(y)
        return np.stack(D), np.stack(M), np.stack(Yl)
    Dtr, Mtr, Ytr = stack(tr_idx); Dva, Mva, Yva = stack(va_idx)
    Gtr = np.stack([run_gate(s) for s in tr_idx]); Gva = np.stack([run_gate(s) for s in va_idx])
    n_desal_on = int(Gtr[:, 0, SLOT_DESAL].sum()); n_pozo_on = int(Gtr[:, 0, SLOT_POZO].sum())
    print(f"[v5.1] action gate: desal-built train runs={n_desal_on}/{len(tr_idx)}  "
          f"new-well-built={n_pozo_on}/{len(tr_idx)}")

    # driver normalization (train stats)
    mu = Dtr.reshape(-1, Din).mean(0); sd = Dtr.reshape(-1, Din).std(0) + 1e-6
    Dtr = (Dtr - mu) / sd; Dva = (Dva - mu) / sd
    # per-alloc-column scale (train median of positive obs) for the loss
    scale = np.array([max(np.nanmedian(Ytr[..., j][Ytr[..., j] > 0]) if np.any(Ytr[..., j] > 0) else 1.0, 1.0)
                      for j in range(n_alloc)], dtype=np.float32)

    dev = "cpu"
    head = LearnedAllocationHead(Din, src_col, src_mask, src_cap, n_alloc, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    scale_t = torch.tensor(scale)
    Dtr_t = torch.tensor(Dtr); Mtr_t = torch.tensor(Mtr); Ytr_t = torch.tensor(Ytr)
    Dva_t = torch.tensor(Dva); Mva_t = torch.tensor(Mva); Yva_t = torch.tensor(Yva)
    Gtr_t = torch.tensor(Gtr); Gva_t = torch.tensor(Gva)   # (n_runs, n_towns, K) bool

    def post(x):   # mask warmup
        return x[:, WARMUP:]

    best = -1e9; best_ep = -1; bad = 0
    epochs = 3 if args.smoke else args.epochs
    bs = 8
    for ep in range(epochs):
        head.train(); perm = np.random.permutation(len(tr_idx))
        tot = 0.0; nb = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            d = Dtr_t[b].to(dev); m = Mtr_t[b].to(dev); y = Ytr_t[b].to(dev)
            g = Gtr_t[b].unsqueeze(1).to(dev)                  # (B,1,n_towns,K) gate
            recon, unmet, met = head(d, m, dyn_mask=g)
            L = alloc_loss(post(recon), post(y), scale_t)
            opt.zero_grad(); L.backward(); opt.step()
            tot += L.item(); nb += 1
        # val: per-source transmission KGE
        head.eval()
        with torch.no_grad():
            recon, _, _ = head(Dva_t.to(dev), Mva_t.to(dev), dyn_mask=Gva_t.unsqueeze(1).to(dev))
            rp = post(recon).cpu().numpy(); yp = post(Yva_t).cpu().numpy()
        kges = [kge(yp[..., j].ravel(), rp[..., j].ravel()) for j in range(n_alloc)]
        mk = float(np.nanmedian(kges))
        if ep % 10 == 0 or args.smoke:
            print(f"  ep{ep:03d} train_loss={tot/max(nb,1):.4f}  val median transmission KGE={mk:.3f}")
        if mk > best:
            best, best_ep, bad = mk, ep, 0
            torch.save({"state_dict": head.state_dict(), "mu": mu, "sd": sd, "scale": scale,
                        "src_col": src_col, "src_mask": src_mask, "src_cap": src_cap,
                        "town_list": town_list, "src_types": src_types, "alloc_cols": alloc_cols,
                        "tourist": tourist, "Din": Din, "hidden": args.hidden,
                        "gate": {"desal_feats": gate_desal_feats, "pozo_feat": gate_pozo_feat,
                                 "slot_desal": SLOT_DESAL, "slot_pozo": SLOT_POZO}},
                       out / "alloc_head_best.pt")
        else:
            bad += 1
            if bad >= args.patience and not args.smoke:
                print(f"  early stop @ ep{ep} (best ep{best_ep})"); break

    # final per-source-type KGE breakdown
    ck = torch.load(out / "alloc_head_best.pt", weights_only=False)
    head.load_state_dict(ck["state_dict"]); head.eval()
    with torch.no_grad():
        recon, _, _ = head(Dva_t.to(dev), Mva_t.to(dev), dyn_mask=Gva_t.unsqueeze(1).to(dev))
        rp = post(recon).cpu().numpy(); yp = post(Yva_t).cpu().numpy()
    by_type = {st: [] for st in src_types}
    j = 0
    for town in town_list:
        for st in src_types:
            if st in towns[town]["srcs"]:
                by_type[st].append(kge(yp[..., j].ravel(), rp[..., j].ravel())); j += 1
    print(f"\n[v5] BEST median transmission KGE = {best:.3f} (ep {best_ep})")
    print("[v5] KGE by source type (val):")
    for st in src_types:
        v = by_type[st]
        if v:
            print(f"    {st:12s} {np.nanmedian(v):6.3f}  (n={len(v)})")
    json.dump({"best_kge": best, "best_ep": best_ep,
               "kge_by_type": {k: float(np.nanmedian(v)) if v else None for k, v in by_type.items()}},
              open(out / "v5_summary.json", "w"), indent=2)
    print(f"[v5] saved -> {out/'alloc_head_best.pt'}")


if __name__ == "__main__":
    main()
