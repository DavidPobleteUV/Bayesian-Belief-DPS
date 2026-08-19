# -*- coding: utf-8 -*-
"""
build_waterfall_preds.py
========================
Toma los pred zarrs nativos del MLP (v2, v3) y produce versiones
"v2.3" / "v3.3" sobrescribiendo los flujos de los transmission links
de fuentes AP (Withdrawal_Node_*) por una asignación determinista en
cascada de precios anclada al pozo predicho:

  well(MLP)  ->  Aduccion  ->  PozoCostero  ->  Desal  ->  Camiones  ->  unmet

Capacidades (L/s):  Aduccion=50, PozoCostero=120, Desal=inf, Camiones=inf
Gates:              Desal y PozoCostero requieren su acción activada en X.
Acuerdo:            NO entra en la cascada (es agrícola; quedaría dominante).

Resultado: results/mlp_preds_<variant>_waterfall.zarr
"""
from __future__ import annotations
import sys, json, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import zarr

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from train_v5_allocation import build_well_caps, build_registry, SEC, WARMUP   # noqa

LPS = 604.8
CAP_LPS = {"Aduccion": 2.0, "PozoCostero": 120.0, "Desal": 1e9, "Camiones": 1e9}
PRIORITY = ["Aduccion", "PozoCostero", "Desal", "Camiones"]

TRUTH = REPO.parent / "WEAP_2_ZARR/results/training_data/merged_iter02_patched/weap_weekly.zarr"
COST_CSV = REPO.parent / "Bayesian-Belief-DPS/data_weap/reference/town_source_cost_mapping.csv"

VARIANTS = [
    ("v2_baseline_AGRmult", "v2_3_waterfall_AGRmult"),
    ("v3_cascade_AGRmult",  "v3_3_waterfall_AGRmult"),
]


def waterfall(need: np.ndarray, well: np.ndarray, src_present: set,
              gates: dict, Tn: int) -> dict:
    rem = np.maximum(need - well, 0.0)
    out = {"Well": np.minimum(well, need)}
    for st in PRIORITY:
        if st in src_present and gates.get(st, True):
            f = np.minimum(rem, CAP_LPS[st] * LPS)
        else:
            f = np.zeros(Tn)
        out[st] = f
        rem = rem - f
    out["unmet"] = rem
    return out


def build_pred_registry(towns, pred_targ_names, pred_feat_names, truth_targ_names):
    """Mapea cada town a (well_col_pred, {src_type: pred_col, ...}, dcol_X).
    Si un link no está en pred.targ, se omite."""
    p_idx = {n: i for i, n in enumerate(pred_targ_names)}
    f_idx = {n: i for i, n in enumerate(pred_feat_names)}
    reg = {}
    for town, T in towns.items():
        srcs = T["srcs"]
        # well: ti del truth -> nombre -> indice en pred
        if "well" not in srcs:
            continue
        well_name = truth_targ_names[srcs["well"]]
        well_pred_col = p_idx.get(well_name)
        if well_pred_col is None:
            continue
        # source-type links en pred
        type_pred_col = {}
        for st in ["Aduccion", "PozoCostero", "Desal", "Camiones", "Acuerdo"]:
            if st in srcs:
                nm = truth_targ_names[srcs[st]]
                if nm in p_idx:
                    type_pred_col[st] = p_idx[nm]
        reg[town] = dict(well_col=well_pred_col, type_col=type_pred_col,
                         dcol=T["dcol"], present=set(type_pred_col.keys()))
    return reg, f_idx


def patch_one_variant(src_zarr_path: Path, dst_zarr_path: Path,
                      towns, truth_targ_names):
    print(f"\n=== {src_zarr_path.name}  ->  {dst_zarr_path.name} ===")
    if dst_zarr_path.exists():
        shutil.rmtree(dst_zarr_path)
    shutil.copytree(src_zarr_path, dst_zarr_path)
    Zp = zarr.open_group(str(dst_zarr_path), mode="r+")
    pred_targ = list(Zp.attrs["target_names"])
    pred_feat = list(Zp.attrs["feature_names"])
    Y = Zp["Y"][:]
    X = Zp["X"][:]
    rids = list(Zp["run_ids"][:])
    Tn = Y.shape[1]

    reg, f_idx = build_pred_registry(towns, pred_targ, pred_feat, truth_targ_names)

    q_desal = [f_idx[n] for n in ["q_desalacion_costera", "q_desalacion_completa"] if n in f_idx]
    q_pozo  = f_idx.get("q_nuevo_pozo_a_5km")

    n_links_changed = 0
    for ri, r in enumerate(rids):
        desal_on = any(np.nanmax(X[ri, :, c]) > 0.5 for c in q_desal)
        pozo_on  = (q_pozo is not None and np.nanmax(X[ri, :, q_pozo]) > 0.5)
        gates = {"Desal": desal_on, "PozoCostero": pozo_on}
        for town, R in reg.items():
            need = X[ri, :, R["dcol"]] * SEC / 0.70   # m³/sem ÷ eficiencia
            well = Y[ri, :, R["well_col"]]
            wf = waterfall(need, well, R["present"], gates, Tn)
            # sobrescribir las columnas existentes en pred Y
            for st, col in R["type_col"].items():
                if st in wf:
                    Y[ri, :, col] = wf[st]
                    n_links_changed += 1

    Zp["Y"][:] = Y
    print(f"  ✓ {len(rids)} runs patchados. Cambios totales en columnas link: {n_links_changed}")


def main():
    # Cargar registro de towns desde el zarr truth
    Zt = zarr.open_group(str(TRUTH), mode="r")
    truth_targ = list(Zt.attrs["target_names"])
    well_caps = build_well_caps(REPO / "data/Q_wells.xlsx")
    cost = pd.read_csv(COST_CSV)
    towns, _, _ = build_registry(Zt, well_caps, cost)
    print(f"Towns detectadas: {len(towns)}")

    for src_name, dst_name in VARIANTS:
        src = REPO / "results" / f"mlp_preds_{src_name}.zarr"
        dst = REPO / "results" / f"mlp_preds_{dst_name}.zarr"
        if not src.exists():
            print(f"  ⚠ {src} no existe, salto.")
            continue
        patch_one_variant(src, dst, towns, truth_targ)


if __name__ == "__main__":
    main()
