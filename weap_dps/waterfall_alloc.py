# -*- coding: utf-8 -*-
"""
waterfall_alloc.py — the `.3` deterministic well-anchored allocation cascade,
applied IN-LOOP to the DPS surface predictions before J4 cost.

Mirrors WEAP_HydroMLP_RecursiveGW/eval_v33_waterfall.py: keep the model's native
WELL flow, then fill each town's residual demand by price priority

    well(native) -> aduccion -> pozo-costero -> desal -> camiones -> (unmet)

Desal, pozo-costero y acuerdo se condicionan a sus banderas de acción (activa en
algún punto del horizonte). Cada fuente respeta su tope físico: aducción 2 L/s,
pozo costero 120 L/s, acuerdo 25 L/s por localidad; desalación y camiones cubren
el resto sin tope efectivo.

El orden ya no está fijo: lo deriva `merit_order()` de las tarifas unitarias, las
mismas con que `cost_calculator` factura (ver esa función).

La cascada sobrescribe, por localidad, las columnas de enlace de transmisión de
`surf_denorm`. `cost_calculator` corre después sin cambios sobre el arreglo
modificado, de modo que J4 refleja la asignación de la cascada.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from weap_dps.config_weap import (TOWN_SOURCE_COST_CSV, TRAIN_ZARR_PATH,
                                  UNIT_COST_BY_SOURCE)

SEC = 604800.0      # s per week
LPS = 604.8         # 1 L/s expressed in m3/week-ish factor used offline (m3/s*SEC)
# physical caps (L/s) -> m3/week via *LPS; desal/camiones effectively uncapped
# Topes físicos (L/s) -> m3/semana vía *LPS. Desal y camiones van sin tope
# efectivo. El acuerdo tiene 25 L/s POR LOCALIDAD (§2.2 de la metodología): son
# arreglos bilaterales, no una tubería compartida.
CAP_LPS = {"Aduccion": 2.0, "PozoCostero": 120.0, "Desal": 1e9,
           "Acuerdo": 25.0, "Camiones": 1e9}
DEMAND_EFFICIENCY = 0.70   # gross need = demand / 0.70 (matches offline)

# Fuentes que participan de la cascada.
#
# El acuerdo estuvo excluido, y se le ponían los enlaces en cero, porque SIN TOPE
# absorbía todo el residuo y anulaba los camiones aljibe —que son la fuente de
# última instancia y la señal de estrés del sistema—. Pero el acuerdo sí tiene
# tope: 25 L/s por localidad. Excluirlo lo dejaba estrictamente dominado: seguía
# costando valor agrícola (el emulador responde a su bandera) sin aportar agua
# urbana ni reducir el costo, de modo que ninguna política racional lo activaba y
# el catálogo quedaba reducido de cuatro acciones a tres. Con su capacidad real
# entra en la cascada como cualquier otra fuente.
CASCADE_SOURCES = ("Aduccion", "PozoCostero", "Desal", "Acuerdo", "Camiones")


def merit_order(unit_costs: dict | None = None,
                sources: tuple[str, ...] = CASCADE_SOURCES) -> list[str]:
    """Orden de despacho por costo unitario CRECIENTE.

    El orden se DERIVA de los precios en vez de fijarse, porque son parámetros
    del análisis: al barrer tarifas para mapear vulnerabilidad, un orden fijo
    seguiría despachando según el mérito antiguo y simularía un despacho que
    ningún operador elegiría —por ejemplo, seguir aduciendo cuando la desalación
    pasó a ser más barata—. Además garantiza que el despacho y `cost_calculator`
    usen los MISMOS precios: si divergen, el DPS optimiza contra un sistema
    incoherente consigo mismo.

    Con las tarifas por omisión el resultado es
    ``["Aduccion", "PozoCostero", "Desal", "Camiones"]``, que es el orden que
    antes estaba escrito a mano; el cambio no altera los resultados vigentes.
    """
    costs = dict(UNIT_COST_BY_SOURCE if unit_costs is None else unit_costs)
    faltan = [s for s in sources if s not in costs]
    if faltan:
        raise ValueError(f"Sin costo unitario para {faltan}: no se puede ordenar "
                         f"el despacho. Disponibles: {sorted(costs)}")
    return sorted(sources, key=lambda s: float(costs[s]))


# Orden por omisión, para compatibilidad con quien importe el símbolo.
PRIORITY = merit_order()


def _build_registry(Z, cost):
    """Trimmed port of train_v5_allocation.build_registry: town -> link names.

    Returns {town: {dem, fict, desal_name, cam_name, adu_name, pozo_name, well_name}}
    using FULL target-name strings (resolved to DPS column indices by the caller).
    """
    targ = list(Z.attrs["target_names"]); feat = list(Z.attrs["feature_names"])
    links = [n for n in targ if "AP_TransmissionLinks" in n and "_to_" in n]
    dem_nodes = sorted({n.split("__", 1)[1].rsplit("_to_", 1)[1] for n in links})
    cost = cost.copy()
    cost["withdrawal_node"] = cost["withdrawal_node"].astype(str).str.strip()
    towns = {}
    for dem in dem_nodes:
        m = re.search(r"_Dem_(\w+)", dem)
        if not m:
            continue
        town = m.group(1)
        if town == "ElManzanoL":
            continue
        dcol_name = "AP_WaterDemand__" + dem
        if dcol_name not in feat:
            continue
        my = [n for n in links if n.endswith("_to_" + dem)]
        names = {}
        for n in my:
            src = n.split("__", 1)[1].rsplit("_to_", 1)[0].replace("Transmission_Link_from_", "")
            if ("APR_" in src or "APU_" in src) and "_Fict_" in src:
                names["well"] = n
            elif src.startswith("Withdrawal_Node_"):
                node = src.replace("Withdrawal_Node_", "")
                row = cost[cost["withdrawal_node"] == node]
                if len(row):
                    names[str(row.iloc[0]["source_type"]).strip()] = n
            elif src.startswith("DemAGRO_SHAC"):
                names["Acuerdo"] = n
        if "well" not in names:
            continue
        towns[town] = dict(dem=dem, dcol_name=dcol_name, names=names)
    return towns


class WaterfallAllocator:
    """Builds the town->link mapping once; applies the cascade per scenario."""

    def __init__(self, surrogate, feature_names, target_names_surf,
                 unit_costs: dict | None = None):
        self.surr = surrogate
        # El registro town->links se lee del MISMO zarr de entrenamiento que usa
        # el resto del DPS. Antes apuntaba a data/weap_weekly.zarr (layout
        # antiguo de 773 runs), que podía tener otro conjunto de enlaces que el
        # dataset con el que se entrenó el modelo en uso.
        Z = zarr.open_group(str(TRAIN_ZARR_PATH), mode="r")
        # Orden de despacho derivado de las tarifas vigentes (ver merit_order).
        self.priority = merit_order(unit_costs)
        self.unit_costs = dict(UNIT_COST_BY_SOURCE if unit_costs is None else unit_costs)
        cost = pd.read_csv(TOWN_SOURCE_COST_CSV)
        reg = _build_registry(Z, cost)

        surf_idx = {n: i for i, n in enumerate(target_names_surf)}
        feat_idx = {n: i for i, n in enumerate(feature_names)}

        # Resolve each town to DPS column indices; keep only what maps cleanly.
        self.towns = {}
        for town, T in reg.items():
            dcol = feat_idx.get(T["dcol_name"])
            well_col = surf_idx.get(T["names"].get("well"))
            if dcol is None or well_col is None:
                continue
            cols = {}
            for st in ["Aduccion", "PozoCostero", "Desal", "Camiones", "Acuerdo"]:
                nm = T["names"].get(st)
                if nm is not None and nm in surf_idx:
                    cols[st] = surf_idx[nm]
            self.towns[town] = dict(dcol=dcol, well_col=well_col, cols=cols)

        # X-denorm params for the demand column (inverse of normalize_x_value).
        self.x_mean = np.asarray(surrogate.x_mean) if surrogate.x_mean is not None else None
        self.x_std = np.asarray(surrogate.x_std) if surrogate.x_std is not None else None
        self.x_methods = surrogate.transform_methods_x_filt
        self.alpha = surrogate.transform_alpha

        # action gating: q_* columns in X (active over horizon -> source available)
        self.q_desal_cols = [feat_idx[c] for c in
                             ["q_desalacion_costera", "q_desalacion_completa"] if c in feat_idx]
        self.q_pozo_col = feat_idx.get("q_nuevo_pozo_a_5km")
        self.q_acuerdo_col = feat_idx.get("q_acuerdo")

    # ── inverse X transform for the demand column ──
    def _denorm_x(self, x_norm: np.ndarray, col: int) -> np.ndarray:
        if self.x_mean is None:
            return x_norm
        unstd = x_norm * (self.x_std[col] if self.x_std[col] > 1e-12 else 1.0) + self.x_mean[col]
        method = "none"
        if self.x_methods is not None and col < len(self.x_methods):
            method = str(self.x_methods[col])
        if method == "log":
            return np.maximum(np.exp(np.clip(unstd, -30, 30)) - self.alpha, 0.0)
        if method == "arcsinh":
            return np.sinh(unstd) * self.alpha
        return unstd

    def apply(self, surf_denorm: np.ndarray, X_used: np.ndarray) -> np.ndarray:
        """Return surf_denorm with desal/aduccion/pozo/camiones overwritten by the
        well-anchored cascade and Acuerdo zeroed, per town. X_used = normalized X
        actually used in the rollout (actions injected); demand & gates read from it.
        """
        out = surf_denorm.copy()
        Tn = out.shape[0]
        # global action gates (step functions -> active if ever >0 over horizon)
        desal_on = any(np.nanmax(X_used[:, c]) > 0 for c in self.q_desal_cols)
        pozo_on = (self.q_pozo_col is not None and np.nanmax(X_used[:, self.q_pozo_col]) > 0)
        acuerdo_on = (self.q_acuerdo_col is not None
                      and np.nanmax(X_used[:, self.q_acuerdo_col]) > 0)
        gates = {"Desal": desal_on, "PozoCostero": pozo_on, "Acuerdo": acuerdo_on}

        for town, R in self.towns.items():
            demand_raw = self._denorm_x(X_used[:, R["dcol"]], R["dcol"])   # m3/s
            need = np.maximum(demand_raw, 0.0) * SEC / DEMAND_EFFICIENCY    # m3/week
            well = np.maximum(out[:, R["well_col"]], 0.0)
            rem = np.maximum(need - well, 0.0)
            for st in self.priority:
                if st not in R["cols"]:
                    continue
                if gates.get(st, True):
                    cap = CAP_LPS[st] * LPS
                    f = np.minimum(rem, cap)
                else:
                    f = np.zeros(Tn)
                out[:, R["cols"][st]] = f
                rem = rem - f
        return out
