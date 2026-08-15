# -*- coding: utf-8 -*-
"""
scenario_builder.py — ensamble de estados del mundo (SOW) para el Robust DPS.

Cada escenario = un X normalizado listo para el rollout. Se parte del template
normalizado (run-0) y se SOBRESCRIBEN las columnas que cambian, normalizándolas
con los scalers del surrogate. Los lags GW iniciales no se tocan: el rollout los
reemplaza con sus propias predicciones a partir del warmup.

Tres incertidumbres, las mismas del diseño experimental de WEAP:

  clima      : series curadas en config_weap.DPS_CLIMATE_RUNS
  población  : 2% / 3% / 5% anual
  uso de suelo agrícola : -50% / -15% / 0% / +20%

Se combinan con un diseño BALANCEADO (build_sow_design), no con un factorial:
9x3x4 serían 108 escenarios y 226 h de optimización. Con 27 se cubren las 12
combinaciones población×área y cada clima aparece 3 veces.

IMPORTANTE: al inyectar el clima hay que recalcular la precipitación ACUMULADA
(PP_acum_*). Es la escala a la que responde el acuífero, y dejarla con los
valores del template hacía que el modelo viera la misma climatología en todos
los escenarios — la dimensión climática del ensamble quedaba muerta.

Devuelve (scenarios: list[np.ndarray], labels: list[str]).
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import (MODEL_REPO, WEEKS_PER_YEAR, TRAIN_ZARR_PATH,
                                  DPS_CLIMATE_RUNS, POP_LEVELS, AREA_LEVELS,
                                  DPS_N_SOW)
from weap_dps.climate_sampler import SUBCUENCAS
from weap_dps.demand_builder import POP_COLUMNS, AREA_COLUMNS, DEMAND_AP_COLUMNS

# Corners heredados (población, área). Se mantienen para reproducir corridas
# antiguas: el ensamble vigente usa el diseño balanceado de SOW, que separa
# población de área en vez de confundirlas. Ver config_weap.DPS_N_SOW.
DEMAND_CORNERS = {
    "HIGH": (0.05, 1.00),
    "MID":  (0.03, 1.00),
    "LOW":  (0.02, 0.50),
}


def build_sow_design(climate_runs: list[int], n_sow: int) -> list[tuple]:
    """Diseño balanceado sobre clima × población × área.

    Recorre el producto población×área con paso coprimo a su tamaño (12), de
    modo que climas consecutivos reciben combinaciones distintas y, al dar la
    vuelta, cada nivel de cada factor queda representado por igual.

    Devuelve [(run_clima, nombre_pob, tasa_pob, nombre_area, mult_area), ...].
    """
    combos = list(itertools.product(POP_LEVELS.items(), AREA_LEVELS.items()))
    paso = 5 if len(combos) % 5 else 7          # coprimo con len(combos)
    out = []
    for i in range(n_sow):
        c = climate_runs[i % len(climate_runs)]
        (pn, pv), (an, av) = combos[(i * paso) % len(combos)]
        out.append((c, pn, pv, an, av))
    return out


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


# Ventanas de precipitación acumulada presentes como features del modelo.
PP_ACUM_WINDOWS = (4, 12, 26, 52, 104)
# Columna de la que se acumulan. Despejado por mínimos cuadrados sobre 5 runs:
# PP_acum_N = suma móvil de N semanas de Precipitation__Subcuenca_Q01, con peso
# exactamente 1.0 en Q01 y 0.0 en las otras cinco (error 0.000% en las cinco
# ventanas). NO es el promedio de la cuenca: usar la media simple da 9% de error.
PP_ACUM_SOURCE = "Precipitation__Subcuenca_Q01"


def _rolling_sum(v: np.ndarray, n: int) -> np.ndarray:
    """Suma de las últimas n semanas; al inicio, de las que haya (parcial)."""
    c = np.cumsum(np.insert(v.astype(float), 0, 0.0))
    out = c[1:].copy()                       # t < n: acumulado parcial
    out[n:] = c[n + 1:] - c[1:-n]
    return out


def _grow_pop(base_series: np.ndarray, rate: float) -> np.ndarray:
    """pop[t] = base_year0 * (1+rate)^año (escalón anual)."""
    T = len(base_series); out = base_series.copy().astype(float)
    base0 = float(base_series[0])
    for y in range(T // WEEKS_PER_YEAR + 1):
        t0 = y * WEEKS_PER_YEAR; t1 = min(t0 + WEEKS_PER_YEAR, T)
        out[t0:t1] = base0 * (1.0 + rate) ** y
    return out


def _scale_demand(base_series: np.ndarray, rate: float) -> np.ndarray:
    """Demanda potable: patrón semanal del PRIMER año, amplificado por (1+rate)^año.

    Antes multiplicaba la serie completa del template por (1+rate)^año, pero esa
    serie YA trae el crecimiento propio del run baseline (2%/año). Las tasas se
    componían: el corner HIGH terminaba creciendo 1.02*1.05-1 = 7.10%/año en vez
    de 5%, MID 5.06% en vez de 3% y LOW 4.04% en vez de 2%.

    Peor aún, quedaba inconsistente con `_grow_pop`, que sí reinicia desde el
    año 0: la población crecía a la tasa pedida y la demanda a otra distinta.
    Ahora ambas usan la misma convención.
    """
    T = len(base_series)
    out = base_series.copy().astype(float)
    y0 = base_series[:WEEKS_PER_YEAR].astype(float)      # estacionalidad base
    for y in range(T // WEEKS_PER_YEAR + 1):
        t0 = y * WEEKS_PER_YEAR
        t1 = min(t0 + WEEKS_PER_YEAR, T)
        if t1 > t0:
            out[t0:t1] = y0[:t1 - t0] * (1.0 + rate) ** y
    return out


def pick_climate_runs(n_climate: int = 5) -> list[int]:
    """N run_ids del ensamble climático.

    Prioridad:
      1. DPS_CLIMATE_RUNS del config — lista curada explícitamente (ver allí el
         porqué: la selección automática mezclaba proyecciones GCM con sequías
         sintéticas extremas y repartía mal el rango).
      2. subset_climate_runs guardado en el zarr reducido.
      3. reparto automático seco→húmedo por precipitación total.
    """
    if DPS_CLIMATE_RUNS:
        runs = [int(r) for r in DPS_CLIMATE_RUNS]
        if n_climate >= len(runs):
            return runs
        # submuestreo uniforme conservando los extremos del rango
        idx = np.linspace(0, len(runs) - 1, n_climate).astype(int)
        return [runs[i] for i in idx]

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
                    n_climate: int = 5, corners: dict | None = None,
                    n_sow: int | None = None) -> tuple[list, list]:
    """Ensamble de estados del mundo.

    Por defecto usa el diseño BALANCEADO sobre clima × población × área
    (config_weap.DPS_N_SOW). Pasando `corners` se recupera el comportamiento
    antiguo (producto climas × corners), solo para reproducir corridas viejas.
    """
    fi = {n: i for i, n in enumerate(feature_names)}
    Z = zarr.open_group(str(TRAIN_ZARR_PATH), mode="r")
    zfeat = list(Z.attrs["feature_names"]); zfi = {n: i for i, n in enumerate(zfeat)}
    zr = np.array(Z["run_ids"][:]).astype(int)
    base_raw = Z["X"][int(np.where(zr == 0)[0][0])]            # run-0 raw (demanda/área/pob base)
    T = template.shape[0]

    climate_runs = pick_climate_runs(n_climate)
    faltan = [c for c in climate_runs if c not in set(zr.tolist())]
    if faltan:
        raise RuntimeError(
            f"Los runs climáticos {faltan} no están en {TRAIN_ZARR_PATH.name} "
            f"(tiene {len(zr)} runs). Si es el subset reducido, regenéralo con "
            f"`python weap_dps/build_train_subset.py` (necesita el zarr completo), "
            f"o apunta al completo con $env:DPS_TRAIN_ZARR.")
    clim_raw = {c: Z["X"][int(np.where(zr == c)[0][0])] for c in climate_runs}

    precip = [f"Precipitation__{s}" for s in SUBCUENCAS]
    temp = [f"Temperature__{s}" for s in SUBCUENCAS]

    if corners is not None:                      # modo heredado
        sow = [(c, cn, pr, f"area{am:.2f}", am)
               for c in climate_runs for cn, (pr, am) in corners.items()]
    else:
        sow = build_sow_design(climate_runs, n_sow or DPS_N_SOW)

    scenarios, labels = [], []
    for c, pname, prate, aname, amult in sow:
        X = template.copy()
        # --- clima: precip/temp del run c ---
        for col in precip + temp:
            if col in fi and col in zfi:
                X[:, fi[col]] = _normalize_col(surrogate, clim_raw[c][:T, zfi[col]], fi[col])
        # --- precipitación ACUMULADA, derivada de la que se acaba de inyectar ---
        # Sin esto quedaban con los valores del template: los 9 escenarios
        # tenían PP_acum_52weeks = 140.4 (el del run 0) cuando en el zarr va
        # de 140 a 274. El acuífero responde a estas escalas, no a la lluvia
        # semanal, así que el clima NO entraba al modelo: el almacenamiento
        # variaba 0.02% entre el clima más seco y el más húmedo, contra 8%
        # en WEAP.
        if PP_ACUM_SOURCE in zfi:
            pp_raw = clim_raw[c][:T, zfi[PP_ACUM_SOURCE]]
            for w in PP_ACUM_WINDOWS:
                col = f"PP_acum_{w}weeks"
                if col in fi:
                    X[:, fi[col]] = _normalize_col(
                        surrogate, _rolling_sum(pp_raw, w), fi[col])
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
        labels.append(f"clim{c}_pop{pname}_area{aname}")
    return scenarios, labels
