# -*- coding: utf-8 -*-
"""
cost_calculator.py — Calcula los 5 objetivos del Pareto (J1..J5) a partir
de las salidas DESNORMALIZADAS del MLP.

Objetivos:
  J1 = min_gw_storage      (maximizar)
  J2 = unmet_ap_total      (minimizar)  [m³ acumulados]
  J3 = agricultural_value  (maximizar)  [CLP acumulados]
  J4 = supply_cost_total   (minimizar)  [CLP acumulados]
  J5 = weeks_in_failure    (minimizar)  [conteo]

Cada función toma los outputs del rollout y devuelve un escalar.
La signature de cada Jx es:  fn(gw_denorm, surf_denorm, target_names, actions_history, mask) -> float
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from weap_dps.config_weap import (
    PRECIO_PALTO_CLP_PER_KG,
    TARIFA_ACUERDO_CLP_PER_M3,
    TOWN_SOURCE_COST_CSV,
    PUMPING_RHO_KG_PER_M3, PUMPING_G_M_PER_S2,
    PUMPING_EFFICIENCY, PUMPING_EXTRA_LIFT_M,
    ENERGY_PRICE_CLP_PER_KWH, J_PER_KWH,
    DISCOUNT_RATE, BASE_YEAR, ANALYSIS_HORIZON_Y, USD_CLP_RATE,
    ACTION_INFRA_PARAMS,
    J5_FAILURE_THRESHOLD_FRAC, WEEKS_PER_YEAR, WARMUP_WEEKS,
)

logger = logging.getLogger(__name__)

# ─── Unidades / conversiones (WEAP) ─────────────────────────────────
SECONDS_PER_WEEK   = 604800     # 7 × 86400 — para convertir m³/s × semana → m³
KG_PER_SHORT_TON   = 907.18474  # 1 short ton (US) = 907.18 kg (NO metric ton)
M3_TO_MM3          = 1e-6       # m³ → Mm³ (millones)


# ─── Lookup de costos por Withdrawal Node / DemAGRO ─────────────────
def _load_cost_lookup() -> dict:
    """
    Carga el CSV town_source_cost_mapping.csv y devuelve dos lookups:
      - by_node:  {str(withdrawal_node): (source_type, unit_cost_clp_m3, town)}
      - by_demagro: {DemAGRO_SHAC_QXX_fict: (source_type, unit_cost, town)}

    El CSV vive en data_weap/reference/.
    """
    if not TOWN_SOURCE_COST_CSV.exists():
        logger.warning("CSV de costos no encontrado: %s", TOWN_SOURCE_COST_CSV)
        return {}
    import pandas as pd
    df = pd.read_csv(TOWN_SOURCE_COST_CSV)
    lookup = {}
    for _, row in df.iterrows():
        key = str(row["withdrawal_node"]).strip()
        lookup[key] = (
            str(row["source_type"]).strip(),
            float(row["unit_cost_clp_m3"]),
            str(row["town"]).strip(),
        )
    return lookup


_COST_LOOKUP_CACHE: dict | None = None


def _get_cost_lookup() -> dict:
    global _COST_LOOKUP_CACHE
    if _COST_LOOKUP_CACHE is None:
        _COST_LOOKUP_CACHE = _load_cost_lookup()
    return _COST_LOOKUP_CACHE


# ─── Helper: extraer fuente y zona del link name ────────────────────
import re

_RE_LINK_SOURCE = re.compile(r"Transmission Link from (.+?) to ")


def _parse_link_source(col_name: str) -> str | None:
    """De 'AP_TransmissionLinks__Transmission Link from APR_Q01_Fict_X to Y'
       devuelve 'APR_Q01_Fict_X'."""
    m = _RE_LINK_SOURCE.search(col_name)
    return m.group(1).strip() if m else None


def _depth_series_for_pozo_group(
    gw_denorm: np.ndarray,
    target_names_gw: list[str],
    source_level2: str,
) -> np.ndarray | None:
    """
    Promedia las series temporales WF_DepthToWater_m de todos los pozos
    bajo `source_level2` (e.g. 'APR_Q01_Fict_JuntaTilama').

    Si no hay match exacto, hace fallback al SHAC (zona Q01..Q09):
    promedia todos los WF_DepthToWater_m del mismo SHAC.

    Returns
    -------
    depth_series (T,) o None si no hay depth en absoluto.
    """
    # 1. Match exacto por source_level2 (ej. APR_Q01_Fict_JuntaTilama)
    cols = [
        i for i, n in enumerate(target_names_gw)
        if "WF_DepthToWater_m" in n and source_level2 in n
    ]
    if cols:
        return np.nanmean(gw_denorm[:, cols], axis=1)

    # 2. Fallback: extraer SHAC (Q01..Q09) y promediar todos los pozos del SHAC
    m = re.search(r"Q(\d{1,2})", source_level2)
    if m is None:
        return None
    shac_tag = f"Q{int(m.group(1)):02d}"
    cols = [
        i for i, n in enumerate(target_names_gw)
        if "WF_DepthToWater_m" in n and shac_tag in n
    ]
    if cols:
        logger.info("    Fallback depth %s (sin pozos directos) → promedio SHAC %s (%d pozos)",
                    source_level2, shac_tag, len(cols))
        return np.nanmean(gw_denorm[:, cols], axis=1)
    return None


def _pumping_cost_per_week(
    flow_m3_per_week: np.ndarray,
    depth_to_water_m: np.ndarray,
) -> np.ndarray:
    """
    Costo eléctrico semana a semana para bombear flow_m3_per_week desde
    profundidad depth_to_water_m + EXTRA_LIFT_M.

    Returns
    -------
    cost_per_week (T,)  CLP
    """
    head = depth_to_water_m + PUMPING_EXTRA_LIFT_M               # m
    energy_J = (flow_m3_per_week * PUMPING_RHO_KG_PER_M3
                * PUMPING_G_M_PER_S2 * head / PUMPING_EFFICIENCY)
    energy_kWh = energy_J / J_PER_KWH
    return energy_kWh * ENERGY_PRICE_CLP_PER_KWH


def _safe_indices(names: Sequence[str], patterns: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if any(p in n for p in patterns)]


# ─── J1: GW storage (maximizar) ──────────────────────────────────────────
def j1_gw_storage(gw_denorm: np.ndarray,
                  target_names: list[str],
                  mask: np.ndarray | None = None,
                  decision_start_week: int = WARMUP_WEEKS) -> float:
    """
    Suma de SHAC_storage_Acuifero_Q01..Q09_MF_m3, agregada como el mínimo
    a lo largo del horizonte de decisión (desde decision_start_week).
    Cuanto mayor el mínimo histórico, mejor.
    """
    cols = _safe_indices(target_names,
                          ["SHAC_storage_Acuifero_Q01_MF_m3",
                           "SHAC_storage_Acuifero_Q02_MF_m3",
                           "SHAC_storage_Acuifero_Q03_MF_m3",
                           "SHAC_storage_Acuifero_Q04_MF_m3",
                           "SHAC_storage_Acuifero_Q05_MF_m3",
                           "SHAC_storage_Acuifero_Q06_MF_m3",
                           "SHAC_storage_Acuifero_Q07_MF_m3",
                           "SHAC_storage_Acuifero_Q08_MF_m3",
                           "SHAC_storage_Acuifero_Q09_MF_m3"])
    if not cols:
        return float("nan")
    series = gw_denorm[decision_start_week:, cols].sum(axis=1)
    return float(np.nanmin(series))


# ─── J2: Unmet demand AP (minimizar) ─────────────────────────────────────
def j2_unmet_ap(surf_denorm: np.ndarray,
                target_names: list[str],
                decision_start_week: int = WARMUP_WEEKS) -> float:
    """
    Suma cumulativa de Demanda No Atendida AP (towns) en m³ a lo largo
    del horizonte.

    Las columnas AP_UnmetDemand están en m³/s (caudal instantáneo);
    multiplicamos por SECONDS_PER_WEEK para convertir a m³/semana antes
    de sumar.

    Clipeamos a 0 antes de sumar: el MLP puede predecir valores ligeramente
    negativos por la inversa del log transform (exp(z) - alpha puede caer
    hasta -0.1 por columna), pero unmet demand físico es siempre ≥ 0.
    """
    cols = _safe_indices(target_names,
                          ["AP_UnmetDemand", "Demanda No Atendida (l_s) - Agua Potable",
                           "Demanda No Atendida AP"])
    if not cols:
        return float("nan")
    # m³/s × 604800 s/semana = m³/semana
    series_m3 = surf_denorm[decision_start_week:, cols] * SECONDS_PER_WEEK
    # Clip negativos a 0 (sin significado físico)
    series_m3 = np.maximum(series_m3, 0.0)
    return float(np.nansum(series_m3))


# ─── J3: Valor agrícola anual (maximizar) ─────────────────────────────────
def j3_agricultural_value(surf_denorm: np.ndarray,
                          target_names: list[str],
                          decision_start_week: int = WARMUP_WEEKS,
                          precio_clp_per_kg: float = PRECIO_PALTO_CLP_PER_KG) -> float:
    """
    Valor agrícola **NPV descontado al 10%** (CLP), año base = BASE_YEAR.

    Mismo patrón que J4: cada año se calcula la producción × precio, se
    descuenta a t=0 y se suma.

      annual_value_t = production_short_ton_t × 907.18 kg × precio
      NPV = Σ_t (annual_value_t / (1 + r)^t)

    Unidad raw: CLP NPV. Para display, dividir por USD_CLP_RATE × 1e6.
    """
    # Solo Palto: aunque el manifest actual solo tiene Palto (12 series:
    # 6 zonas × Goteo/Microaspersion), explicitamos para evitar contar
    # otros cultivos si se agregan en el futuro.
    cols = [i for i, n in enumerate(target_names)
            if "AGR_AnnualCropProduction" in n and ("Palto" in n or "palto" in n)]
    if not cols:
        return float("nan")
    # Producción está replicada weekly → tomamos el mean por año (= valor anual)
    # Clip negativos del MLP (sin significado físico).
    series = np.maximum(surf_denorm[decision_start_week:, cols], 0.0)
    T = series.shape[0]
    n_years = T // WEEKS_PER_YEAR

    # Producción por año (suma sobre las 12 series agrícolas)
    annual_short_ton = np.array([
        np.nansum(np.nanmean(series[y*WEEKS_PER_YEAR:(y+1)*WEEKS_PER_YEAR], axis=0))
        for y in range(n_years)
    ])
    annual_clp = annual_short_ton * KG_PER_SHORT_TON * precio_clp_per_kg

    # Descontar a NPV (t=0 = BASE_YEAR)
    r = DISCOUNT_RATE
    t = np.arange(n_years, dtype=float)
    npv = float(np.nansum(annual_clp / (1.0 + r) ** t))

    # EAC para display/log (no afecta optimización)
    crf = _annuity_factor(r, ANALYSIS_HORIZON_Y)
    eac = npv * crf
    logger.info("J3 — Valor agrícola NPV (r=%.0f%%): %.3e CLP   (EAC = %.3e CLP/año, %.2f MUSD/año)",
                r * 100, npv, eac, eac / USD_CLP_RATE / 1e6)
    return npv


# ─── J4: Costo de abastecimiento anualizado (NPV + EAC, minimizar) ──────
def _annuity_factor(rate: float, n: int) -> float:
    """CRF (capital recovery factor): NPV × CRF = EAC."""
    return rate / (1.0 - (1.0 + rate) ** -n)


def _discount_yearly(yearly_values: np.ndarray, rate: float) -> float:
    """NPV de una serie anual {CF_0, CF_1, ..., CF_{N-1}}."""
    t = np.arange(len(yearly_values), dtype=float)
    return float(np.nansum(yearly_values / (1.0 + rate) ** t))


def _detect_first_activation_year(action_series: np.ndarray) -> int | None:
    """
    Primer año (índice 0-based desde BASE_YEAR) donde el binario == 1.
    None si nunca se activa.
    """
    actives = np.where(action_series > 0.5)[0]
    return int(actives[0]) if len(actives) else None


def _weekly_to_yearly(weekly_series: np.ndarray, n_years: int) -> np.ndarray:
    """Suma semanal → suma anual. Shape: (n_years,)."""
    yearly = np.array([
        np.nansum(weekly_series[y * WEEKS_PER_YEAR : (y + 1) * WEEKS_PER_YEAR])
        for y in range(n_years)
    ])
    return yearly


def j4_supply_cost(
    surf_denorm: np.ndarray,
    target_names_surf: list[str],
    gw_denorm: np.ndarray,
    target_names_gw: list[str],
    decision_start_week: int = WARMUP_WEEKS,
    actions_history: np.ndarray | None = None,
    action_names_order: list[str] | None = None,
    return_breakdown: bool = False,
) -> float | dict:
    """
    Costo CLP acumulado de abastecer a las towns durante el horizonte de
    decisión. Suma sobre todos los transmission links que terminan en una
    town, clasificados por tipo de fuente.

    Tres tipos de fuente:

    1. **Pozos regulares** (`APR_Q*_Fict_<town>`):
       costo eléctrico de bombeo:
         cost = Σ_t flow_t × ρ × g × (depth_to_water_t + 10) / η / J_per_kWh × price
       donde `depth_to_water_t` es el promedio temporal de `WF_DepthToWater_m`
       sobre los pozos del town (variables MLP en gw_denorm).

    2. **Withdrawal Nodes** (de town_source_cost_mapping.csv):
       Aduccion (500), Camiones (8000), Desal (1500), PozoCostero (1200)
       — costo fijo por m³.

    3. **Acuerdo** (`DemAGRO_SHAC_Q*_fict`): 3500 CLP/m³ (fijo).

    Returns
    -------
    Total CLP acumulado durante el horizonte (decision_start_week..fin).
    """
    if surf_denorm is None or target_names_surf is None:
        return float("nan")

    lookup = _get_cost_lookup()

    # Cuántos años de decisión hay en el horizonte
    horizon_weeks = surf_denorm.shape[0] - decision_start_week
    n_years = horizon_weeks // WEEKS_PER_YEAR

    # ── Acumuladores POR AÑO Y POR FUENTE ──────────────────────────────
    # Cada source_type: array (n_years,) con CLP/año (sin descontar aún)
    yearly_opex_by_type = {
        "PozosRegulares": np.zeros(n_years),
        "Aduccion":       np.zeros(n_years),
        "Camiones":       np.zeros(n_years),
        "Desal":          np.zeros(n_years),
        "PozoCostero":    np.zeros(n_years),
        "Acuerdo":        np.zeros(n_years),
    }
    n_links = {k: 0 for k in yearly_opex_by_type}
    n_links["unknown"] = 0

    for j, name in enumerate(target_names_surf):
        if "Transmission Link from" not in name:
            continue
        source = _parse_link_source(name)
        if source is None:
            continue

        flow_per_week = surf_denorm[decision_start_week:, j]  # m³/semana (per usuario)

        # ── Tipo 1: Pozos regulares (APR/APU)_Q*_Fict_<town> ───────────
        if (source.startswith("APR_") or source.startswith("APU_")) and "_Fict_" in source:
            depth_series = _depth_series_for_pozo_group(
                gw_denorm, target_names_gw, source_level2=source,
            )
            if depth_series is None:
                n_links["unknown"] += 1
                continue
            depth_decision = depth_series[decision_start_week:]
            cost_per_week = _pumping_cost_per_week(flow_per_week, depth_decision)
            yearly_cost = _weekly_to_yearly(cost_per_week, n_years)
            yearly_opex_by_type["PozosRegulares"] += yearly_cost
            n_links["PozosRegulares"] += 1

        # ── Tipo 2: Withdrawal Node N (CSV: Aduccion/Camiones/Desal/PozoCostero) ─
        elif source.startswith("Withdrawal Node"):
            node = source.replace("Withdrawal Node", "").strip()
            if node in lookup:
                src_type, unit_cost, _town = lookup[node]
                yearly_vol = _weekly_to_yearly(flow_per_week, n_years)  # m³/año
                yearly_cost = yearly_vol * unit_cost
                if src_type in yearly_opex_by_type:
                    yearly_opex_by_type[src_type] += yearly_cost
                    n_links[src_type] += 1
                else:
                    logger.warning("Tipo desconocido en CSV: %s", src_type)
                    n_links["unknown"] += 1
            else:
                n_links["unknown"] += 1

        # ── Tipo 3: Acuerdo (DemAGRO_SHAC_*_fict) ──────────────────────
        elif source.startswith("DemAGRO_SHAC"):
            if source in lookup:
                _src_type, unit_cost, _town = lookup[source]
            else:
                unit_cost = TARIFA_ACUERDO_CLP_PER_M3
            yearly_vol = _weekly_to_yearly(flow_per_week, n_years)
            yearly_cost = yearly_vol * unit_cost
            yearly_opex_by_type["Acuerdo"] += yearly_cost
            n_links["Acuerdo"] += 1

        else:
            n_links["unknown"] += 1

    # ── Descontar OPEX a NPV (año base = BASE_YEAR, decisión = t=0) ────
    r = DISCOUNT_RATE
    npv_opex_by_type = {
        k: _discount_yearly(v, r) for k, v in yearly_opex_by_type.items()
    }
    npv_opex_total = float(sum(npv_opex_by_type.values()))

    # ── CAPEX: detectar año de activación de cada acción ───────────────
    npv_capex_by_action = {}
    capex_events = []
    if actions_history is not None and action_names_order is not None:
        for i, name in enumerate(action_names_order):
            if name not in ACTION_INFRA_PARAMS:
                continue
            params = ACTION_INFRA_PARAMS[name]
            capex_clp = params["capex_clp"]
            if capex_clp <= 0:
                continue
            # actions_history[year, i] — buscar primera activación
            activation_year = _detect_first_activation_year(actions_history[:, i])
            if activation_year is None:
                continue
            # CAPEX se paga `construction_lead_years` antes de la activación
            lead = params["construction_lead_years"]
            t_capex = max(0, activation_year - lead)
            disc_capex = capex_clp / (1.0 + r) ** t_capex
            npv_capex_by_action[name] = disc_capex
            capex_events.append({
                "action": name,
                "activation_year_idx": activation_year,
                "activation_year_calendar": BASE_YEAR + activation_year,
                "capex_year_idx": t_capex,
                "capex_year_calendar": BASE_YEAR + t_capex,
                "capex_clp_raw": capex_clp,
                "capex_clp_npv": disc_capex,
            })

    npv_capex_total = float(sum(npv_capex_by_action.values()))
    npv_total = npv_opex_total + npv_capex_total

    # ── EAC (Equivalent Annual Cost) ───────────────────────────────────
    crf = _annuity_factor(r, ANALYSIS_HORIZON_Y)
    eac_total = npv_total * crf

    # ── Reporte ────────────────────────────────────────────────────────
    logger.info("J4 — Costo anualizado (descuento r=%.0f%%, base year=%d, N=%d años):",
                r * 100, BASE_YEAR, ANALYSIS_HORIZON_Y)
    logger.info("")
    logger.info("    OPEX desglose por fuente (NPV en CLP):")
    logger.info("    %-16s  %-14s  %-9s  %s",
                "Fuente", "NPV CLP", "% OPEX", "n_links")
    logger.info("    %s", "-" * 60)
    npv_opex_for_pct = npv_opex_total if npv_opex_total > 0 else 1.0
    for k, v in sorted(npv_opex_by_type.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * v / npv_opex_for_pct
        logger.info("    %-16s  %.3e     %5.1f%%     %d",
                    k, v, pct, n_links[k])
    logger.info("    %s", "-" * 60)
    logger.info("    %-16s  %.3e", "NPV OPEX", npv_opex_total)
    logger.info("")
    if capex_events:
        logger.info("    CAPEX events (cada acción activada):")
        for e in capex_events:
            logger.info("      %s  →  activa en %d, CAPEX pagado en %d (raw=%.2eM CLP, NPV=%.2eM CLP)",
                        e["action"], e["activation_year_calendar"],
                        e["capex_year_calendar"],
                        e["capex_clp_raw"] / 1e6, e["capex_clp_npv"] / 1e6)
        logger.info("    %-16s  %.3e", "NPV CAPEX", npv_capex_total)
    else:
        logger.info("    CAPEX events: 0 (ninguna acción nueva activada o todas con CAPEX=0)")
    logger.info("")
    logger.info("    %-16s  %.3e CLP", "NPV TOTAL", npv_total)
    logger.info("    %-16s  %.3e CLP/año  (= %.2f MUSD/año a %.0f CLP/USD)",
                "EAC", eac_total, eac_total / USD_CLP_RATE / 1e6, USD_CLP_RATE)

    if n_links["unknown"]:
        logger.info("    unknown links no clasificados = %d", n_links["unknown"])

    if return_breakdown:
        return {
            "npv_total":            npv_total,
            "npv_opex":             npv_opex_total,
            "npv_capex":            npv_capex_total,
            "eac_total":            eac_total,
            "eac_musd_per_year":    eac_total / USD_CLP_RATE / 1e6,
            "npv_opex_by_type":     npv_opex_by_type,
            "npv_capex_by_action":  npv_capex_by_action,
            "capex_events":         capex_events,
            "n_links":              n_links,
        }
    # Default: NSGA-II minimiza el NPV total
    return float(npv_total)


# ─── J5: Semanas en falla (minimizar) ────────────────────────────────────
def j5_weeks_in_failure(surf_denorm: np.ndarray,
                        target_names: list[str],
                        decision_start_week: int = WARMUP_WEEKS,
                        threshold_frac: float = J5_FAILURE_THRESHOLD_FRAC,
                        abs_threshold_m3_week: float | None = None) -> float:
    """
    Cuenta semanas con falla de AP usando dos criterios alternativos:

      1. Si la demanda AP está disponible como TARGET, usa ratio Unmet/Demand.
      2. Si no (la demanda es INPUT en este modelo), usa umbral absoluto en m³/semana:
         "semana en falla = sum_unmet_AP_m3 > abs_threshold_m3_week".

    Las columnas AP_UnmetDemand están en m³/s. Antes de comparar con el
    threshold (m³/semana) convertimos:  m³/s × 604800 = m³/semana.

    Default abs_threshold_m3_week = 100 m³/semana.
    """
    unmet_cols = _safe_indices(target_names,
                                ["AP_UnmetDemand", "Demanda No Atendida AP",
                                 "Unmet demand"])
    if not unmet_cols:
        return float("nan")

    # m³/s → m³/semana (clip a 0 para evitar negativos del MLP inverso log)
    unmet_m3 = surf_denorm[decision_start_week:, unmet_cols] * SECONDS_PER_WEEK
    unmet_m3 = np.maximum(unmet_m3, 0.0)
    unmet_per_t = unmet_m3.sum(axis=1)

    dem_cols = _safe_indices(target_names,
                              ["AP_WaterDemand", "Demanda de Agua - Agua Potable"])
    if dem_cols:
        # Modo ratio (demand también en m³/s → convertir)
        dem_m3 = surf_denorm[decision_start_week:, dem_cols] * SECONDS_PER_WEEK
        dem_per_t = dem_m3.sum(axis=1)
        safe_dem = np.where(dem_per_t > 1e-9, dem_per_t, np.nan)
        ratio = unmet_per_t / safe_dem
        return float(np.nansum(ratio > threshold_frac))

    # Modo umbral absoluto (m³/semana)
    threshold = abs_threshold_m3_week if abs_threshold_m3_week is not None else 100.0
    return float(np.nansum(unmet_per_t > threshold))


# ─── Wrapper: calcula los 5 ──────────────────────────────────────────────
def compute_objectives(gw_denorm: np.ndarray,
                       surf_denorm: np.ndarray,
                       target_names_gw: list[str],
                       target_names_surf: list[str],
                       actions_history: np.ndarray,
                       action_names_order: list[str],
                       decision_start_week: int = WARMUP_WEEKS) -> dict[str, float]:
    return {
        "J1_gw_storage":    j1_gw_storage(gw_denorm, target_names_gw, decision_start_week=decision_start_week),
        "J2_unmet_ap":      j2_unmet_ap(surf_denorm, target_names_surf, decision_start_week=decision_start_week),
        "J3_agri_value":    j3_agricultural_value(surf_denorm, target_names_surf, decision_start_week=decision_start_week),
        "J4_supply_cost":   j4_supply_cost(surf_denorm=surf_denorm,
                                            target_names_surf=target_names_surf,
                                            gw_denorm=gw_denorm,
                                            target_names_gw=target_names_gw,
                                            decision_start_week=decision_start_week,
                                            actions_history=actions_history,
                                            action_names_order=action_names_order),
        "J5_weeks_failure": j5_weeks_in_failure(surf_denorm, target_names_surf, decision_start_week=decision_start_week),
    }
