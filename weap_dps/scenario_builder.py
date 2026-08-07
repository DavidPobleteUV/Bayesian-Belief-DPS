# -*- coding: utf-8 -*-
"""
scenario_builder.py — construye el ensamble climate × demand para Robust DPS.

Cada escenario = un X normalizado (1872, n_x) listo para el rollout. Se parte del
template normalizado (run-0) y se SOBRESCRIBEN solo las columnas que cambian
(precip/temp/pob/demanda/área), normalizándolas con los scalers del surrogate.
Así NO se tocan los lags GW iniciales (recursión).

  climate: N realizaciones (dry→wet por precip total, filtrando P=0)
  demand : corners pob×área   (HIGH 5%/1.0, MID 2%/1.0, LOW 2%/0.50)

Devuelve (scenarios: list[np.ndarray], labels: list[str]).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import MODEL_REPO, WEEKS_PER_YEAR, TRAIN_ZARR_PATH
from weap_dps.climate_sampler import SUBCUENCAS
from weap_dps.demand_builder import POP_COLUMNS, AREA_COLUMNS, DEMAND_AP_COLUMNS

# corners de demanda: (pop_growth, area_mult)
DEMAND_CORNERS = {
    "HIGH": (0.05, 1.00),   # mucha población, sin reducción de área
    "MID":  (0.02, 1.00),
    "LOW":  (0.02, 0.50),   # poca población, -50% área
}


def _normalize_col(surr, raw_series: np.ndarray, col_idx: int) -> np.ndarray:
    """Inversa de denorm: transform (log/arcsinh/none) + z-score, por columna."""
    method = "none"
    if surr.transform_methods_x_filt is not None and col_idx < len(surr.transform_methods_x_filt):
        method = str(surr.transform_methods_x_filt[col_idx])
    a = surr.transform_alpha
    if method == "log":
        tr = np.log(np.maximum(raw_series + a, 1e-12))
    elif method == "arcsinh":
        tr = np.arcsinh(raw_series / a)
    else:
        tr = raw_series.astype(float)
    mean = float(surr.x_mean[col_idx]); std = float(surr.x_std[col_idx])
    return (tr - mean) / (std if std > 1e-12 else 1.0)


def _grow_pop(base_series: np.ndarray, rate: float) -> np.ndarray:
    """pop[t] = base_year0 * (1+rate)^año (escalón anual)."""
    T = len(base_series); out = base_series.copy().astype(float)
    base0 = float(base_series[0])
    for y in range(T // WEEKS_PER_YEAR + 1):
        t0 = y * WEEKS_PER_YEAR; t1 = min(t0 + WEEKS_PER_YEAR, T)
        out[t0:t1] = base0 * (1.0 + rate) ** y
    return out


def _scale_demand(base_series: np.ndarray, rate: float) -> np.ndarray:
    """Demanda potable: conserva el patrón semanal del template, lo amplifica por (1+rate)^año."""
    T = len(base_series); out = base_series.copy().astype(float)
    for y in range(T // WEEKS_PER_YEAR + 1):
        t0 = y * WEEKS_PER_YEAR; t1 = min(t0 + WEEKS_PER_YEAR, T)
        out[t0:t1] = base_series[t0:t1] * (1.0 + rate) ** y
    return out


def pick_climate_runs(n_climate: int = 5) -> list[int]:
    """N run_ids spread dry→wet por precipitación total (filtrando P=0)."""
    Z = zarr.open_group(str(TRAIN_ZARR_PATH), mode="r")

    # Si el zarr es un SUBCONJUNTO (build_train_subset.py), los climas ya vienen
    # pre-seleccionados sobre el dataset COMPLETO y guardados en attrs. Hay que
    # usar esa lista: re-elegir aquí daría un ensamble distinto, porque el
    # subset solo tiene unos pocos runs (y el baseline se colaría como "clima").
    sub = Z.attrs.get("subset_climate_runs")
    if sub:
        sub = [int(x) for x in sub]                      # ya ordenados seco→húmedo
        if n_climate >= len(sub):
            return sub
        idx = np.linspace(0, len(sub) - 1, n_climate).astype(int)
        return [sub[i] for i in idx]

    feat = list(Z.attrs["feature_names"]); rids = np.array(Z["run_ids"][:]).astype(int)
    pcols = [feat.index(f"Precipitation__{s}") for s in SUBCUENCAS if f"Precipitation__{s}" in feat]
    val = np.where(rids >= 0)[0]
    try:                                   # leer SOLO columnas precip (rápido)
        pp = Z["X"].oindex[:, :, pcols]    # (nruns, T, 6)
        tot = np.nansum(pp[val], axis=(1, 2))
    except Exception:                      # fallback por-run
        tot = np.array([float(np.nansum(Z["X"][i][:, pcols])) for i in val])
    ok = tot > 0
    val, tot = val[ok], tot[ok]
    order = np.argsort(tot)
    pick = np.linspace(0, len(order) - 1, n_climate).astype(int)
    return [int(rids[val[order[p]]]) for p in pick]


def build_scenarios(surrogate, feature_names: list[str], template: np.ndarray,
                    n_climate: int = 5, corners: dict | None = None) -> tuple[list, list]:
    corners = corners or DEMAND_CORNERS
    fi = {n: i for i, n in enumerate(feature_names)}
    Z = zarr.open_group(str(TRAIN_ZARR_PATH), mode="r")
    zfeat = list(Z.attrs["feature_names"]); zfi = {n: i for i, n in enumerate(zfeat)}
    zr = np.array(Z["run_ids"][:]).astype(int)
    base_raw = Z["X"][int(np.where(zr == 0)[0][0])]            # run-0 raw (demanda/área/pob base)
    T = template.shape[0]

    climate_runs = pick_climate_runs(n_climate)
    clim_raw = {c: Z["X"][int(np.where(zr == c)[0][0])] for c in climate_runs}

    precip = [f"Precipitation__{s}" for s in SUBCUENCAS]
    temp = [f"Temperature__{s}" for s in SUBCUENCAS]

    scenarios, labels = [], []
    for c in climate_runs:
        for cname, (prate, amult) in corners.items():
            X = template.copy()
            # --- clima: precip/temp del run c ---
            for col in precip + temp:
                if col in fi and col in zfi:
                    X[:, fi[col]] = _normalize_col(surrogate, clim_raw[c][:T, zfi[col]], fi[col])
            # --- población (cap) y demanda potable: crecen con prate ---
            for col in POP_COLUMNS:
                if col in fi and col in zfi:
                    X[:, fi[col]] = _normalize_col(surrogate, _grow_pop(base_raw[:T, zfi[col]], prate), fi[col])
            for col in DEMAND_AP_COLUMNS:
                if col in fi and col in zfi:
                    X[:, fi[col]] = _normalize_col(surrogate, _scale_demand(base_raw[:T, zfi[col]], prate), fi[col])
            # --- área de riego: multiplicador constante ---
            for col in AREA_COLUMNS:
                if col in fi and col in zfi:
                    X[:, fi[col]] = _normalize_col(surrogate, base_raw[:T, zfi[col]] * amult, fi[col])
            scenarios.append(X.astype(np.float32))
            labels.append(f"clim{c}_{cname}(pop{prate:.0%},area{amult:.2f})")
    return scenarios, labels
