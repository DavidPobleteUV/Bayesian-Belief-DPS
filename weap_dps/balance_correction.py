# -*- coding: utf-8 -*-
"""
balance_correction.py — Restaura el balance de agua potable del surrogate.

En WEAP, el agua contabilizada del sistema de agua potable es una función casi
determinista de la demanda:

    suministro + déficit  =  k · demanda,      k = 1.7083

y ese k es notablemente estable entre runs (p10 1.649, p90 1.762, CV 2.6% sobre
los 113 runs de test). El factor supera 1 porque el requerimiento incluye
pérdidas de distribución que `AP_WaterDemand` (demanda neta) no contiene.

El surrogate NO reproduce esa identidad: cierra el balance en 1.435 y con cinco
veces más dispersión (p10 1.205, p90 1.603). Entrega 0.826 del suministro
observado, y ese déficit es el origen común de varios sesgos que se venían
parchando por separado — el volumen de respaldo subestimado, los camiones bajos
y el factor de calibración de J4.

Esta corrección impone la identidad: escala el suministro predicho para que el
balance se cumpla, a resolución ANUAL (que es la del descuento de J4 y mucho más
estable que la semanal, donde un respaldo casi nulo dispara el factor).

Efecto medido sobre los 113 runs de test:

    error de J4   25.2%  ->   7.6%      (p75: 39.4% -> 15.9%)
    sesgo de J4   0.748  ->  0.946
    suministro    0.826  ->  1.048

LIMITACIÓN CONOCIDA: el escalado es uniforme entre fuentes, así que los pozos
propios —que el surrogate ya predice bien (0.984), porque su caudal lo fija el
derecho de agua y la profundidad de la napa, no la asignación— quedan inflados
a 1.283. A 11 CLP/m³ eso es despreciable en J4 y no afecta a J1 (que usa el
almacenamiento del acuífero, no los links), pero SÍ sesga el reparto por fuente
que se reporte. Se probó la variante que corrige solo las fuentes de respaldo y
deja los pozos intactos: preserva el reparto pero deja J4 en 15.1% de error, el
doble. Se eligió la precisión del objetivo sobre la del diagnóstico.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# (suministro + déficit) / demanda medido en WEAP sobre los 113 runs de test.
# Si cambia el modelo WEAP o el conjunto de pueblos, hay que recalcularlo:
#   scratchpad/valida_balance.py
K_BALANCE = 1.7083

# Techo del factor. Con resolución anual el factor típico es ~1.2-1.3; un año
# con suministro predicho casi nulo podría disparar la corrección, y multiplicar
# por 50 un residuo numérico no reconstruye información que no está.
MAX_FACTOR = 3.0

WEEKS_PER_YEAR = 52
SECONDS_PER_WEEK = 604800.0


def apply_balance_correction(surf_denorm: np.ndarray,
                             link_idx: list[int],
                             unmet_idx: list[int],
                             demand_m3s: np.ndarray,
                             decision_start_week: int,
                             k: float = K_BALANCE) -> np.ndarray:
    """Escala los links de suministro para que se cumpla S + U = k·D por año.

    Parameters
    ----------
    surf_denorm : (T, n_surf) predicciones de superficie DESNORMALIZADAS.
    link_idx    : columnas de AP_TransmissionLinks (m³/semana).
    unmet_idx   : columnas de AP_UnmetDemand (m³/s).
    demand_m3s  : (T, n_towns) demanda AP desnormalizada desde X (m³/s).
    decision_start_week : desde dónde se evalúan los objetivos.

    Returns
    -------
    Copia de surf_denorm con los links corregidos. El déficit NO se toca: es la
    base de J2, J51 y J52, y alterarlo cambiaría los objetivos que se optimizan
    en vez de corregir el sesgo del suministro.
    """
    if not link_idx or demand_m3s is None or len(demand_m3s) == 0:
        return surf_denorm

    # ── Guarda de unidades ───────────────────────────────────────────────────
    # `demand_m3s` DEBE venir en m³/s crudos. Es un error fácil de cometer:
    # el zarr de predicciones de evaluate_recursive guarda Y desnormalizado
    # pero X NORMALIZADO, de modo que pasarle su columna AP_WaterDemand entrega
    # z-scores. Con z-scores el residuo (k·D − U) se divide por un S incoherente
    # y el factor satura en MAX_FACTOR para todos los años, inflando el costo
    # sin que nada falle de forma visible.
    d_arr = np.asarray(demand_m3s, dtype=float)
    if np.nanmin(d_arr) < -1e-9:
        raise ValueError(
            "demand_m3s tiene valores negativos: parece normalizado (z-scores) "
            "en vez de m³/s crudos. Toma la demanda del zarr de entrenamiento "
            f"(X crudo), no del zarr de predicciones. min={np.nanmin(d_arr):.4f}")

    out = surf_denorm.copy()
    t0 = decision_start_week
    S = np.maximum(np.nan_to_num(out[t0:, link_idx]), 0.0)          # (T', n_links)
    U = np.maximum(np.nan_to_num(out[t0:, unmet_idx]), 0.0).sum(1) * SECONDS_PER_WEEK
    D = np.maximum(np.nan_to_num(np.asarray(demand_m3s)[t0:]), 0.0).sum(1) * SECONDS_PER_WEEK

    n = min(S.shape[0], U.shape[0], D.shape[0])
    n_y = n // WEEKS_PER_YEAR
    if n_y < 1:
        return surf_denorm

    m = n_y * WEEKS_PER_YEAR
    s_y = S[:m].reshape(n_y, WEEKS_PER_YEAR, -1).sum(axis=1).sum(axis=1)   # (n_y,)
    u_y = U[:m].reshape(n_y, WEEKS_PER_YEAR).sum(axis=1)
    d_y = D[:m].reshape(n_y, WEEKS_PER_YEAR).sum(axis=1)

    # factor >= 1: la corrección solo AÑADE el agua que falta para cerrar el
    # balance. Si el surrogate ya entrega de más, no se le recorta: eso seria
    # empeorar una prediccion que no tiene el sesgo que estamos corrigiendo.
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(s_y > 1e-9, (k * d_y - u_y) / s_y, 1.0)
    f = np.clip(np.nan_to_num(f, nan=1.0, posinf=1.0), 1.0, MAX_FACTOR)

    fw = np.repeat(f, WEEKS_PER_YEAR)                    # (m,) factor por semana
    idx = np.asarray(link_idx, dtype=int)
    out[t0:t0 + m, idx] = S[:m] * fw[:, None]
    return out
