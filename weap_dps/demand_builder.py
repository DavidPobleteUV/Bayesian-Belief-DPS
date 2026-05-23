# -*- coding: utf-8 -*-
"""
demand_builder.py — Construye series de demanda (poblacional y agrícola)
para alimentar al MLP en cada escenario DPS.

Lógica:
  - Población: pop[t] = pop_base_2015 × (1 + growth_rate)^(year - 2015)
  - Áreas:     area[t] = area_base × multiplier (constante en el horizonte)

Las series resultantes se inyectan en las columnas correspondientes del input X.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np

logger = logging.getLogger(__name__)

# Columnas del manifest del MLP
POP_COLUMNS = [
    "AP_Poblacion__APR_Q01_Dem_JuntaTilama (cap)",
    "AP_Poblacion__APR_Q05_Dem_LosCondores (cap)",
    "AP_Poblacion__APR_Q06_Dem_ElManzanoLoClaudio (cap)",
    "AP_Poblacion__APR_Q06_Dem_Guanguali (cap)",
    "AP_Poblacion__APR_Q09_Dem_ElEsfuerzo (cap)",
    "AP_Poblacion__APR_Q09_Dem_LosMaquis (cap)",
    "AP_Poblacion__APR_Q09_Dem_Quilimari (cap)",
    "AP_Poblacion__APU_Q09_Dem_Pichidangui (cap)",
]

AREA_COLUMNS = [
    f"AGR_Area__Agricola_Q0{i}\\Frutales\\Palto\\{tech}"
    for i in range(1, 7) for tech in ("Goteo", "Microaspersion")
]

DEMAND_AP_COLUMNS = [
    "AP_WaterDemand__APR_Q09_Dem_ElEsfuerzo",
    "AP_WaterDemand__APR_Q09_Dem_Quilimari",
    "AP_WaterDemand__APU_Q09_Dem_Pichidangui",
]

WEEKS_PER_YEAR = 52


def apply_population_growth(X: np.ndarray, feat_names: list[str],
                             pop_base: Mapping[str, float],
                             growth_rate: float,
                             start_year: int = 2014) -> np.ndarray:
    """
    Aplica crecimiento poblacional exponencial sobre las 8 columnas
    de población. pop_base[col_name] es el valor en start_year.

    Las series son annual (las 52 weeks de cada año tienen el mismo valor).
    """
    X = X.copy()
    T = X.shape[0]
    n_years = T // WEEKS_PER_YEAR

    for col in POP_COLUMNS:
        if col not in feat_names:
            continue
        idx = feat_names.index(col)
        base = pop_base.get(col, X[0, idx])  # default: valor inicial del template
        for y in range(n_years):
            mult = (1.0 + growth_rate) ** y
            t0 = y * WEEKS_PER_YEAR
            t1 = min(t0 + WEEKS_PER_YEAR, T)
            X[t0:t1, idx] = base * mult
    return X


def apply_area_multiplier(X: np.ndarray, feat_names: list[str],
                          area_base: Mapping[str, float],
                          multiplier: float) -> np.ndarray:
    """
    Aplica un multiplicador constante a las 12 columnas de área agrícola.
    multiplier ∈ {1.0, 0.85, 0.50} típicamente.
    """
    X = X.copy()
    for col in AREA_COLUMNS:
        if col not in feat_names:
            continue
        idx = feat_names.index(col)
        base = area_base.get(col, X[0, idx])
        X[:, idx] = base * multiplier
    return X


def scale_ap_demand_with_population(X: np.ndarray, feat_names: list[str],
                                     growth_rate: float,
                                     start_year: int = 2014) -> np.ndarray:
    """
    Las 3 columnas weekly de AP_WaterDemand también escalan con la población.
    Mantiene el patrón estacional original (del template) y solo lo amplifica
    por el factor de crecimiento.
    """
    X = X.copy()
    T = X.shape[0]
    n_years = T // WEEKS_PER_YEAR
    for col in DEMAND_AP_COLUMNS:
        if col not in feat_names:
            continue
        idx = feat_names.index(col)
        original = X[:, idx].copy()       # patrón semanal del template
        for y in range(n_years):
            mult = (1.0 + growth_rate) ** y
            t0 = y * WEEKS_PER_YEAR
            t1 = min(t0 + WEEKS_PER_YEAR, T)
            X[t0:t1, idx] = original[t0:t1] * mult
    return X


def extract_base_population(X_template: np.ndarray,
                             feat_names: list[str]) -> dict[str, float]:
    """Lee del primer timestep las poblaciones base (asume year 0 = 2014)."""
    return {col: float(X_template[0, feat_names.index(col)])
            for col in POP_COLUMNS if col in feat_names}


def extract_base_areas(X_template: np.ndarray,
                       feat_names: list[str]) -> dict[str, float]:
    """Lee del primer timestep las áreas agrícolas base."""
    return {col: float(X_template[0, feat_names.index(col)])
            for col in AREA_COLUMNS if col in feat_names}
