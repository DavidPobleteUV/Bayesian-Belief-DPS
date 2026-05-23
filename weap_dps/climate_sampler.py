# -*- coding: utf-8 -*-
"""
climate_sampler.py — Extrae series weekly de precip+temp por subcuenca
desde el zarr merged y las prepara como escenarios climáticos para DPS.

Cada escenario es un dict con arrays (T,) por subcuenca:
    {
      "precip": {"Subcuenca_Q01": np.array, ..., "Subcuenca_Q06": np.array},
      "tmean":  {"Subcuenca_Q01": np.array, ...},
      "gcm":    "MPI-ESM1-2-LR",
      "ssp":    "ssp585",
      "source_run_id": 0,
    }
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import MODEL_REPO, CLIMATE_DIR, TOTAL_WEEKS_MLP, GCM_LIST

logger = logging.getLogger(__name__)

SUBCUENCAS = [f"Subcuenca_Q0{i}" for i in range(1, 7)]


def extract_climate_template(zarr_path: Path = None,
                             baseline_run_id: int = 0) -> dict:
    """
    Extrae series weekly de precip y temp para todas las subcuencas desde
    UN run baseline del zarr merged. Sirve como template para escenarios
    sintéticos.
    """
    zarr_path = zarr_path or (MODEL_REPO / "data" / "weap_weekly.zarr")
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr no encontrado: {zarr_path}")

    z = zarr.open_group(str(zarr_path), mode="r")
    feat_names = list(z.attrs.get("feature_names", []))
    run_ids = z["run_ids"][:]
    idx = np.where(run_ids == baseline_run_id)[0]
    if len(idx) == 0:
        raise ValueError(f"run_id={baseline_run_id} no está en el zarr")
    slot = int(idx[0])

    X = z["X"][slot]  # (T, n_feat) sin normalizar
    out_precip = {}
    out_tmean = {}
    for sc in SUBCUENCAS:
        pp_col = f"Precipitation__{sc}"
        tm_col = f"Temperature__{sc}"
        if pp_col in feat_names:
            out_precip[sc] = X[:, feat_names.index(pp_col)].astype(np.float32)
        if tm_col in feat_names:
            out_tmean[sc] = X[:, feat_names.index(tm_col)].astype(np.float32)

    return {
        "precip": out_precip,
        "tmean":  out_tmean,
        "gcm":    "TEMPLATE",
        "ssp":    "BASELINE",
        "source_run_id": baseline_run_id,
    }


def list_climate_runs(zarr_path: Path = None) -> list[dict]:
    """
    Devuelve, para cada GCM × SSP del catálogo, una lista de run_ids del zarr
    que corresponden a ese GCM (mirando la matriz factorial / LHS si está
    disponible). Por ahora retorna lista vacía — se completará en una
    versión posterior con el manifest de runs.
    """
    # TODO: cuando tengamos un manifest run_id → (GCM, SSP), filtrar acá.
    return []


def apply_scenario_to_X(X: np.ndarray, feat_names: list[str],
                        scenario: dict) -> np.ndarray:
    """
    Sobrescribe las columnas de Precipitation__Subcuenca_* y
    Temperature__Subcuenca_* en X con las series del escenario.

    NOTA: X debe estar SIN normalizar; la normalización se aplica después.
    """
    X = X.copy()
    for sc, series in scenario.get("precip", {}).items():
        col = f"Precipitation__{sc}"
        if col in feat_names:
            X[:, feat_names.index(col)] = series[:X.shape[0]]
    for sc, series in scenario.get("tmean", {}).items():
        col = f"Temperature__{sc}"
        if col in feat_names:
            X[:, feat_names.index(col)] = series[:X.shape[0]]
    return X


def save_scenarios_to_disk(scenarios: list[dict], out_dir: Path = None) -> None:
    out_dir = out_dir or CLIMATE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sc in enumerate(scenarios):
        fname = out_dir / f"scenario_{i:03d}_{sc['gcm']}_{sc['ssp']}.npz"
        np.savez_compressed(fname,
                            precip={k: v for k, v in sc["precip"].items()},
                            tmean={k: v for k, v in sc["tmean"].items()},
                            gcm=sc["gcm"], ssp=sc["ssp"])
        logger.info("Saved %s", fname)
