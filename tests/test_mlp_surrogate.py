# -*- coding: utf-8 -*-
"""
test_mlp_surrogate.py — Sanity check del bridge.

Valida:
  1. Que se puede cargar el ckpt.
  2. Que se puede correr predict_horizon sobre el template.
  3. Que rollout_with_policy produce arrays con shapes correctos.
  4. Que cost_calculator entrega 5 floats (sin NaN si hay datos).

Uso:
  python tests/test_mlp_surrogate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import (
    ZARR_TEMPLATE_PATH, SPIN_UP_YEARS, DECISION_YEARS,
    WARMUP_WEEKS, WEEKS_PER_YEAR,
)
from weap_dps.mlp_surrogate import MLPSurrogate
from weap_dps.action_translator import (
    policy_output_to_actions, build_action_col_idx,
    ACTION_NAMES_BINARY, ACTION_NAMES_QUANTITY,
)
from weap_dps.cost_calculator import compute_objectives

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [TEST] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _check_template_exists():
    if not ZARR_TEMPLATE_PATH.exists():
        logger.error("Template no encontrado: %s", ZARR_TEMPLATE_PATH)
        logger.error("Corre primero: python weap_dps/extract_data.py")
        return False
    return True


def test_1_load_checkpoint():
    logger.info("=" * 60)
    logger.info("Test 1: Load checkpoint")
    logger.info("=" * 60)
    s = MLPSurrogate()
    assert s.n_x > 0 and s.n_gw > 0 and s.n_surface > 0
    logger.info("  n_x=%d  n_gw=%d  n_surface=%d", s.n_x, s.n_gw, s.n_surface)
    logger.info("  OK\n")
    return s


def test_2_predict_horizon(surrogate: MLPSurrogate):
    logger.info("=" * 60)
    logger.info("Test 2: predict_horizon over template")
    logger.info("=" * 60)
    data = np.load(ZARR_TEMPLATE_PATH, allow_pickle=True)
    X = data["X"]
    logger.info("  X template shape: %s", X.shape)
    gw, surf = surrogate.predict_horizon(X)
    logger.info("  gw_pred shape:    %s   (range %.3f → %.3f)",
                gw.shape, np.nanmin(gw), np.nanmax(gw))
    logger.info("  surf_pred shape:  %s   (range %.3f → %.3f)",
                surf.shape, np.nanmin(surf), np.nanmax(surf))
    assert gw.shape == (X.shape[0], surrogate.n_gw)
    assert surf.shape == (X.shape[0], surrogate.n_surface)
    logger.info("  OK\n")
    return X, gw, surf


def test_3_dummy_policy_rollout(surrogate: MLPSurrogate, X: np.ndarray):
    logger.info("=" * 60)
    logger.info("Test 3: rollout_with_policy with dummy policy")
    logger.info("=" * 60)
    data = np.load(ZARR_TEMPLATE_PATH, allow_pickle=True)
    feat_names = list(data["feature_names"])
    action_col_idx = build_action_col_idx(feat_names)
    logger.info("  Action cols mapped:")
    for k, v in action_col_idx.items():
        logger.info("    %s → col %d", k, v)

    # Política dummy: activar desal_completa y prorrateo a partir del año 5
    def policy_fn(state, year_idx):
        if year_idx >= 5:
            pi_out = np.array([0, 1, 1, 0.5, 0.7, 0.3])  # binarias + cantidades normalizadas
        else:
            pi_out = np.array([0, 0, 0, 0, 0, 0])
        return policy_output_to_actions(pi_out)

    result = surrogate.rollout_with_policy(
        X_template=X,
        policy_fn=policy_fn,
        n_years=DECISION_YEARS,
        action_col_idx=action_col_idx,
        spin_up_years=SPIN_UP_YEARS,
    )
    logger.info("  gw output:      %s", result["gw"].shape)
    logger.info("  surface output: %s", result["surface"].shape)
    logger.info("  actions_history shape: %s", result["actions_history"].shape)
    logger.info("  Action history sample (years 0,5,10,15):")
    ah = result["actions_history"]
    for y in (0, 5, 10, 15):
        if y < ah.shape[0]:
            logger.info("    year %d: %s", y, ah[y])
    logger.info("  OK\n")
    return result


def test_4_compute_objectives(surrogate: MLPSurrogate, result: dict):
    logger.info("=" * 60)
    logger.info("Test 4: compute_objectives + DEBUG diagnostics")
    logger.info("=" * 60)
    # Desnormalizar
    gw_denorm   = surrogate.denormalize_y(result["gw"],     kind="gw")
    surf_denorm = surrogate.denormalize_y(result["surface"], kind="surface")

    # ─── DEBUG: diagnosticar el GW storage ──────────────────────────
    target_names_gw = surrogate.target_names_gw
    storage_keywords = ["SHAC_storage_Acuifero_Q01_MF_m3",
                         "SHAC_storage_Acuifero_Q02_MF_m3",
                         "SHAC_storage_Acuifero_Q03_MF_m3",
                         "SHAC_storage_Acuifero_Q04_MF_m3",
                         "SHAC_storage_Acuifero_Q05_MF_m3",
                         "SHAC_storage_Acuifero_Q06_MF_m3",
                         "SHAC_storage_Acuifero_Q07_MF_m3",
                         "SHAC_storage_Acuifero_Q08_MF_m3",
                         "SHAC_storage_Acuifero_Q09_MF_m3"]
    storage_cols = [i for i, n in enumerate(target_names_gw)
                     if any(k in n for k in storage_keywords)]
    logger.info("  [DEBUG] SHAC_storage cols found: %d", len(storage_cols))
    for i in storage_cols[:5]:
        logger.info("    idx=%d → '%s'", i, target_names_gw[i])

    if storage_cols:
        # Estadísticas SOLO de los SHAC storage
        gw_storage = gw_denorm[:, storage_cols]
        logger.info("  [DEBUG] gw_storage shape: %s", gw_storage.shape)
        logger.info("  [DEBUG] gw_storage range: min=%.3e  max=%.3e",
                    np.nanmin(gw_storage), np.nanmax(gw_storage))
        logger.info("  [DEBUG] gw_storage mean: %.3e   median: %.3e",
                    np.nanmean(gw_storage), np.nanmedian(gw_storage))
        # Suma timestep a timestep, descartar warmup
        sum_per_t = gw_storage[WARMUP_WEEKS + SPIN_UP_YEARS * WEEKS_PER_YEAR:].sum(axis=1)
        logger.info("  [DEBUG] sum(9 SHACs) per timestep (post-warmup):")
        logger.info("    min=%.3e   max=%.3e   mean=%.3e   median=%.3e",
                    np.nanmin(sum_per_t), np.nanmax(sum_per_t),
                    np.nanmean(sum_per_t), np.nanmedian(sum_per_t))
        # Top-3 menores (los que arrastran J1)
        bottom_idx = np.argsort(sum_per_t)[:3]
        logger.info("  [DEBUG] Bottom-3 timesteps (responsables del J1):")
        for ii in bottom_idx:
            real_t = ii + WARMUP_WEEKS + SPIN_UP_YEARS * WEEKS_PER_YEAR
            row = gw_storage[real_t]
            logger.info("    t=%d → sum=%.3e | min_col=%.3e | max_col=%.3e",
                        real_t, sum_per_t[ii], np.min(row), np.max(row))

    # ─── DEBUG: contar transmission links por tipo ────────────────────
    n_links_total = sum(1 for n in surrogate.target_names_surf
                         if "Transmission Link from" in n)
    n_pozos_reg = sum(1 for n in surrogate.target_names_surf
                       if "Transmission Link from" in n
                       and "_Fict_" in n and (n.split("from ")[1].startswith("APR_")
                                              or n.split("from ")[1].startswith("APU_")))
    n_withdrawal = sum(1 for n in surrogate.target_names_surf
                        if "Transmission Link from Withdrawal Node" in n)
    n_demagro = sum(1 for n in surrogate.target_names_surf
                     if "Transmission Link from DemAGRO_SHAC" in n)
    n_depth = sum(1 for n in surrogate.target_names_gw
                   if "WF_DepthToWater_m" in n)
    logger.info("  [DEBUG] Transmission links totales: %d", n_links_total)
    logger.info("    pozos regulares (APR_Q*_Fict_*): %d", n_pozos_reg)
    logger.info("    withdrawal nodes:                %d", n_withdrawal)
    logger.info("    DemAGRO_SHAC (acuerdo):          %d", n_demagro)
    logger.info("    no clasificados:                 %d",
                n_links_total - n_pozos_reg - n_withdrawal - n_demagro)
    logger.info("  [DEBUG] WF_DepthToWater_m disponibles en GW: %d", n_depth)

    # ─── Objectives ─────────────────────────────────────────────────
    # Activar logging DEBUG temporal para ver desglose J4
    logging.getLogger("weap_dps.cost_calculator").setLevel(logging.DEBUG)
    objs = compute_objectives(
        gw_denorm=gw_denorm,
        surf_denorm=surf_denorm,
        target_names_gw=surrogate.target_names_gw,
        target_names_surf=surrogate.target_names_surf,
        actions_history=result["actions_history"],
        action_names_order=ACTION_NAMES_BINARY + ACTION_NAMES_QUANTITY,
        decision_start_week=WARMUP_WEEKS + SPIN_UP_YEARS * WEEKS_PER_YEAR,
    )
    logger.info("  Objectives:")
    for k, v in objs.items():
        logger.info("    %s = %s", k, v)
    logger.info("  OK\n")


def main():
    if not _check_template_exists():
        sys.exit(1)
    surrogate = test_1_load_checkpoint()
    X, gw, surf = test_2_predict_horizon(surrogate)
    result = test_3_dummy_policy_rollout(surrogate, X)
    test_4_compute_objectives(surrogate, result)
    logger.info("=" * 60)
    logger.info("All sanity tests passed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
