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

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from weap_dps.config_weap import (TOWN_SOURCE_COST_CSV, TRAIN_ZARR_PATH,
                                  UNIT_COST_BY_SOURCE)

logger = logging.getLogger(__name__)

SEC = 604800.0      # s per week
LPS = 604.8         # 1 L/s expressed in m3/week-ish factor used offline (m3/s*SEC)
# physical caps (L/s) -> m3/week via *LPS; desal/camiones effectively uncapped
# Topes físicos (L/s) -> m3/semana vía *LPS. Desal y camiones van sin tope
# efectivo.
#
# El acuerdo tiene 25 L/s POR SHAC, no por localidad: el cupo lo fija el acuífero
# que cede el agua, y las localidades que extraen del mismo SHAC lo COMPARTEN.
# Ver `SHARED_CAP_SOURCES` y `apply`. Aplicarlo por enlace —como se hacía antes—
# multiplicaba la capacidad real: Q09 abastece cuatro localidades, de modo que su
# tope efectivo pasaba de 25 a 100 L/s, y ahí está el 94 % de la población. El
# efecto medido sobre el frente era un volumen cedido de 33 Mm³ contra los 7.9
# que entrega WMMaS2.
CAP_LPS = {"Aduccion": 2.0, "PozoCostero": 120.0, "Desal": 1e9,
           "Acuerdo": 25.0, "Camiones": 1e9}
DEMAND_EFFICIENCY = 0.70   # gross need = demand / 0.70 (matches offline)

# Fuentes que participan de la cascada. El acuerdo ENTRA, con su tope real.
#
# Su tope es de 25 L/s POR SHAC y lo COMPARTEN las localidades que extraen del
# mismo acuífero agrícola (ver SHARED_CAP_SOURCES y `apply`). Aplicarlo por enlace
# —como se hacía antes— multiplicaba la capacidad: Q09 abastece cuatro
# localidades, de modo que su tope efectivo era 100 y no 25 L/s, y ahí está el
# 94 % de la población.
#
# Volumen cedido en el frente, contra las 30 corridas de WMMaS2 donde el acuerdo
# opera (24.46 Mm³, tasa 0.904 Mm³ por año encendido):
#
#     tope por enlace          33.0 Mm³   +35 %
#     tope por SHAC            23.2 Mm³    -5 %   <- vigente
#     fuera de la cascada      27.5 Mm³   +12 %   (predicción nativa del emulador)
#
# La opción vigente es la más fiel de las tres. Se evaluó sacarlo de la cascada
# para conservar la predicción nativa —que el criterio 3 puntúa con razón 0.965,
# la mejor de todas las fuentes— y resulta peor que el tope por SHAC, además de
# dejar al acuerdo fuera del orden de mérito y por tanto fuera del análisis de
# umbrales de precio (§4.7).
CASCADE_SOURCES = ("Aduccion", "PozoCostero", "Desal", "Acuerdo", "Camiones")

# Fuentes cuyo tope es COMPARTIDO por un grupo de localidades en vez de aplicarse
# a cada enlace por separado. El cupo del grupo se reparte a prorrata del agua que
# le falta a cada localidad tras las fuentes más baratas, sin prioridad entre
# ellas: ninguna localidad tiene preferencia sobre otra del mismo SHAC.
SHARED_CAP_SOURCES = ("Acuerdo",)

# Localidades costeras que la desaladora costera puede abastecer y cuyo nombre no
# lleva el prefijo Q09. Hoy esta vacio —las cuatro costeras son Q09— y existe
# para no tener que tocar la logica si se agrega una.
COSTERAS_EXTRA: set[str] = set()


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

        # Grupo de tope compartido por localidad. El acuerdo se limita POR SHAC,
        # de modo que las localidades que extraen del mismo acuífero agrícola
        # comparten los 25 L/s. El grupo se lee del propio nombre del enlace
        # (DemAGRO_SHAC_QXX_fict), no de una tabla aparte, para que no pueda
        # quedar desincronizado del registro.
        self.grupo_compartido = {}
        for town, T in reg.items():
            if town not in self.towns:
                continue
            nm = T["names"].get("Acuerdo")
            g = re.search(r"(DemAGRO_SHAC_Q\d+)", nm or "")
            self.grupo_compartido[town] = g.group(1) if g else town
        _g = {}
        for t, k in self.grupo_compartido.items():
            _g.setdefault(k, []).append(t)
        logger.info("Tope compartido del acuerdo: %d SHAC -> %s",
                    len(_g), {k.replace("DemAGRO_SHAC_", ""): sorted(v)
                              for k, v in sorted(_g.items())})

        # Gating de acciones: columnas q_* en X. Las dos desaladoras se mantienen
        # SEPARADAS porque no tienen el mismo alcance (ver `_gates`): la costera
        # abastece solo a las localidades de Q09, la completa al sistema entero.
        self.q_desal_costera_col = feat_idx.get("q_desalacion_costera")
        self.q_desal_completa_col = feat_idx.get("q_desalacion_completa")
        self.q_desal_cols = [c for c in (self.q_desal_costera_col,
                                         self.q_desal_completa_col) if c is not None]
        self.q_pozo_col = feat_idx.get("q_nuevo_pozo_a_5km")
        self.q_acuerdo_col = feat_idx.get("q_acuerdo")

    @staticmethod
    def _es_q09(town: str) -> bool:
        """La desaladora costera solo llega a las localidades del sector Q09."""
        return "Q09" in town or town in COSTERAS_EXTRA

    def _gates(self, X_used: np.ndarray, town: str) -> dict:
        """Disponibilidad de cada fuente SEMANA A SEMANA para una localidad.

        El gate se evalua por paso de tiempo, no una vez sobre todo el horizonte.
        La version anterior usaba `np.nanmax(X_used[:, c]) > 0`, es decir, bastaba
        que la accion se encendiera alguna vez para que la cascada despachara la
        obra desde la primera semana: una desaladora activada en 2045 entregaba
        agua desde 2027 mientras el CAPEX se cobraba en la fecha correcta. Medido
        sobre el frente, eso ocurria en el 37.8 % de las semanas para la
        desalacion, el 42.0 % para el pozo costero y el 67.5 % para el acuerdo, y
        subestimaba J4 en un 5.4 % mediano (hasta 51.6 %) ademas de distorsionar
        por completo el reparto entre fuentes.
        """
        Tn = X_used.shape[0]
        def on(col):
            return (np.nan_to_num(X_used[:, col]) > 0 if col is not None
                    else np.zeros(Tn, bool))
        # La costera solo habilita a Q09; la completa habilita a cualquiera.
        desal = on(self.q_desal_completa_col)
        if self._es_q09(town):
            desal = desal | on(self.q_desal_costera_col)
        return {"Desal": desal,
                "PozoCostero": on(self.q_pozo_col),
                "Acuerdo": on(self.q_acuerdo_col)}

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
        """Return surf_denorm with the cascade allocation written into the links.

        Recorre las fuentes por ORDEN DE MÉRITO (bucle externo) y las localidades
        por dentro, en vez de al revés. El orden importa: las fuentes de tope
        COMPARTIDO —hoy solo el acuerdo, ver SHARED_CAP_SOURCES— necesitan ver el
        faltante de todas las localidades del grupo a la vez para poder repartir
        el cupo a prorrata. Con el bucle invertido, cada localidad consumía el
        tope completo como si fuera suyo.
        """
        out = surf_denorm.copy()
        Tn = out.shape[0]

        # Faltante por localidad tras el pozo propio, que es el ancla de la cascada.
        rem, gates = {}, {}
        for town, R in self.towns.items():
            demand_raw = self._denorm_x(X_used[:, R["dcol"]], R["dcol"])   # m3/s
            need = np.maximum(demand_raw, 0.0) * SEC / DEMAND_EFFICIENCY   # m3/week
            well = np.maximum(out[:, R["well_col"]], 0.0)
            rem[town] = np.maximum(need - well, 0.0)
            gates[town] = self._gates(X_used, town)

        for st in self.priority:
            cap = CAP_LPS[st] * LPS
            elegibles = [t for t, R in self.towns.items() if st in R["cols"]]
            if not elegibles:
                continue

            if st in SHARED_CAP_SOURCES:
                # Un cupo por grupo (SHAC), repartido en proporción al faltante.
                grupos = {}
                for t in elegibles:
                    grupos.setdefault(self.grupo_compartido.get(t, t), []).append(t)
                for _, miembros in grupos.items():
                    activos = {t: np.where(gates[t].get(st, np.ones(Tn, bool)),
                                           rem[t], 0.0) for t in miembros}
                    total = np.sum([activos[t] for t in miembros], axis=0)
                    entrega = np.minimum(total, cap)
                    # Sin faltante no hay reparto; el where evita 0/0.
                    escala = np.divide(entrega, total, out=np.zeros(Tn),
                                       where=total > 1e-12)
                    for t in miembros:
                        f = activos[t] * escala
                        out[:, self.towns[t]["cols"][st]] = f
                        rem[t] = rem[t] - f
            else:
                for t in elegibles:
                    g = gates[t].get(st)
                    if g is None:
                        g = np.ones(Tn, bool)
                    f = np.where(g, np.minimum(rem[t], cap), 0.0)
                    out[:, self.towns[t]["cols"][st]] = f
                    rem[t] = rem[t] - f
        return out
