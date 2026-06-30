# -*- coding: utf-8 -*-
"""
config_weap.py — Constantes del case study Quilimari para Standard DPS.

Cualquier ajuste de hiperparámetro de la simulación, tarifas, áreas base o
horizonte debe pasar por aquí. NO hardcodear valores en otros módulos.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data_weap"
# Checkpoint puede sobreescribirse por entorno (DPS_CKPT) para correr variantes
# de modelo (v2/v3) sin tocar el default. Default = best_model.ckpt.
CKPT_PATH    = Path(os.environ["DPS_CKPT"]) if os.environ.get("DPS_CKPT") else DATA_DIR / "best_model.ckpt"

# Flag .3: si DPS_WATERFALL=1, J4 usa la cascada determinista well-anclada
# (well nativo -> aduccion -> pozo-costero -> desal -> camiones) para derivar
# desal/camiones como déficit con prioridad de precio, en vez de las predicciones
# nativas del modelo. Ver weap_dps/waterfall_alloc.py.
DPS_WATERFALL = os.environ.get("DPS_WATERFALL", "0") == "1"
MANIFEST_PATH = DATA_DIR / "manifest_inputs.csv"
SCALERS_PATH  = DATA_DIR / "scalers_weap.npz"
TRANSFORM_PARAMS_PATH = DATA_DIR / "transform_params_weap.npz"
CLIMATE_DIR   = DATA_DIR / "climate_base"
ZARR_TEMPLATE_PATH = DATA_DIR / "X_template.npz"   # 1 run baseline para reusar
RESULTS_DIR  = PROJECT_ROOT / "runs_weap"

# Repo del modelo (para `extract_data.py`)
MODEL_REPO = PROJECT_ROOT.parent / "WEAP_HydroMLP_RecursiveGW"

# ─── Horizonte temporal ────────────────────────────────────────────────────
# El MLP fue entrenado sobre 1872 weeks (2014-04-02 → 2050-03-26, calendario WEAP).
TOTAL_WEEKS_MLP = 1872

# Timeline real (week0 = 2014-04-02, año hidrológico inicia 2-abril):
#   warmup   : wk   0..103  = 2014-04 .. 2016-04   (2 años, recursión MLP)
#   spin-up  : wk 104..675  = 2016-04 .. 2027-04   (11 años, datos hist.)
#   decisión : wk 676..1871 = 2027-04 .. 2050-03   (23 años-agua que caben en 1872 wk)
# OJO: las decisiones DPS arrancan en 2027-04 (NO 2025). decision_start_week
#   = WARMUP + SPIN_UP*52 = 104 + 572 = 676 -> 2027-04-02 (verificado vs time[].
#   Por eso start_year del export = 2027 y BASE_YEAR = 2027.
SPIN_UP_YEARS = 11        # 2016-04 .. 2027-04
DECISION_YEARS = 26       # nominal; solo 23 años-agua caben en el horizonte de 1872 wk
WARMUP_WEEKS = 104        # primeros 2 años, fixed por el MLP
WEEKS_PER_YEAR = 52

# Cuántas semanas usar en total = warmup + spin-up + decision
N_WEEKS_HORIZON = min(TOTAL_WEEKS_MLP, WARMUP_WEEKS + (SPIN_UP_YEARS + DECISION_YEARS) * WEEKS_PER_YEAR)

# ─── Espacio de acción ─────────────────────────────────────────────────────
# Las 5 acciones son BINARIAS PURAS (on/off). En los datos de entrenamiento del
# MLP cada acción aparece SOLO con un único valor de cantidad (q canónico), así
# que NO hay decisión continua: al activarse una binaria se inyecta su q fijo.
# Tratarlas como continuas haría que el MLP extrapole a cantidades nunca vistas.
ACTION_NAMES_BINARY = [
    "act_desalacion_costera",
    "act_desalacion_completa",
    "act_prorrateo_shac",
    "act_prorrateo_cuenca",
    "act_nuevo_pozo_a_5km",
]

# q canónico (valor único observado en el zarr de entrenamiento) que se inyecta
# cuando la binaria correspondiente está activa.
CANONICAL_Q = {
    "q_desalacion_costera":  0.1,    # planta pequeña
    "q_desalacion_completa": 0.3,    # planta grande
    "q_prorrateo_shac":      0.85,
    "q_prorrateo_cuenca":    0.7,
    "q_nuevo_pozo_a_5km":    0.12,
}

# Nombres de las columnas q en el MISMO orden que ACTION_NAMES_BINARY.
ACTION_NAMES_QUANTITY = [
    "q_desalacion_costera",
    "q_desalacion_completa",
    "q_prorrateo_shac",
    "q_prorrateo_cuenca",
    "q_nuevo_pozo_a_5km",
]

# Compat: bounds degenerados (lo=hi=canónico). Ya NO se usan como rango de
# decisión, pero se mantienen por si algún script viejo los referencia.
Q_BOUNDS = {q: (v, v) for q, v in CANONICAL_Q.items()}

# ─── Tarifas y precios ─────────────────────────────────────────────────────
PRECIO_PALTO_CLP_PER_KG = 2200.0                    # CLP/kg (Poblete et al. 2025)

# Path al CSV con tarifas por transmission link (town × source × node)
REFERENCE_DIR = DATA_DIR / "reference"
TOWN_SOURCE_COST_CSV = REFERENCE_DIR / "town_source_cost_mapping.csv"

# Costos unitarios por tipo de fuente (CLP/m³), PARÁMETROS tuneables.
# El CSV town_source_cost_mapping.csv solo MAPEA qué nodo es cada tipo; el
# PRECIO sale de aquí → sensibilidad/robustez sin re-simular ni regenerar el zarr.
UNIT_COST_BY_SOURCE = {
    "Camiones":    8000.0,
    "Desal":       2300.0,
    "Aduccion":    1200.0,
    "PozoCostero": 1500.0,
    "Acuerdo":     2500.0,
}
# Compat: fallback de Acuerdo (= UNIT_COST_BY_SOURCE["Acuerdo"])
TARIFA_ACUERDO_CLP_PER_M3 = UNIT_COST_BY_SOURCE["Acuerdo"]

# ─── Parámetros para costo eléctrico de bombeo (pozos regulares) ───────────
# Aplica SOLO a transmission links desde APR_Q*_Fict_<town> (pozos baseline
# de cada utility). PozoCostero, Desal, Aduccion, Camiones, Acuerdo usan
# UNIT_COST_BY_SOURCE (arriba).
PUMPING_RHO_KG_PER_M3   = 1000.0    # densidad del agua
PUMPING_G_M_PER_S2      = 9.81      # gravedad
PUMPING_EFFICIENCY      = 0.80      # eficiencia del sistema (80%)
PUMPING_EXTRA_LIFT_M    = 10.0      # carga adicional sobre depth_to_water
ENERGY_PRICE_CLP_PER_KWH = 300.0    # CLP/kWh (alineado con WEAP_2_ZARR)
J_PER_KWH               = 3.6e6     # conversión J → kWh

# ─── Anualización de costos (NPV + EAC) ────────────────────────────────────
# Todos los flujos de caja (CAPEX y OPEX) se descuentan al año t=0 = BASE_YEAR.
# NSGA-II minimiza el NPV total. EAC se reporta solo para interpretación.
# NOTA: el descuento es por ÍNDICE de año-decisión (año 0 = primera decisión),
#   así que BASE_YEAR solo fija las etiquetas calendario de display; cambiarlo
#   NO altera los valores NPV (las fronteras del baseline siguen válidas).
DISCOUNT_RATE       = 0.10          # 10% anual
BASE_YEAR           = 2027          # t = 0 = primera decisión DPS (2027-04, verificado)
ANALYSIS_HORIZON_Y  = 23            # años-agua de análisis (2027-2050)
USD_CLP_RATE        = 950.0         # tipo de cambio para display USD/CLP

# CAPEX + parámetros temporales por acción nueva.
# Camiones, Aduccion, baseline Desal, baseline PozoCostero, baseline Pozos
# Regulares NO tienen CAPEX (son fuentes existentes / contratos por m³).
ACTION_INFRA_PARAMS = {
    "act_desalacion_costera": {
        "capex_clp":                5.0e9,    # 5.000 millones CLP (~$5.3M USD)
        "construction_lead_years":  2,
        "lifetime_years":           25,
    },
    "act_desalacion_completa": {
        "capex_clp":                12.0e9,   # 12.000 millones CLP (~$12.6M USD)
        "construction_lead_years":  3,
        "lifetime_years":           25,
    },
    "act_nuevo_pozo_a_5km": {
        "capex_clp":                1.5e9,    # 1.500 millones CLP (~$1.6M USD)
        "construction_lead_years":  1,
        "lifetime_years":           30,
    },
    "act_prorrateo_shac": {
        "capex_clp":                0.0,      # administrativo, sin infra
        "construction_lead_years":  0,
        "lifetime_years":           25,
    },
    "act_prorrateo_cuenca": {
        "capex_clp":                0.0,      # administrativo, sin infra
        "construction_lead_years":  0,
        "lifetime_years":           25,
    },
}

# ─── Umbrales / criterios de objetivos ─────────────────────────────────────
J5_FAILURE_THRESHOLD_FRAC = 0.10   # semana en falla si Unmet/Demand > 10%

# Calibración de J4 (costo): factor = E[costo_obs]/E[costo_pred] (agregado, 118
# test runs). Re-derivado por variante para los modelos limpios via
# calibrate_j4_waterfall.py (el viejo 1.22 era del modelo antiguo). El costo es
# ~95% camiones. NO altera el orden de Pareto (factor constante), solo el valor
# absoluto reportado. Sobreescribible por entorno DPS_J4_CAL (lo setea el runner
# por variante):  v2=1.149  v3=1.032  v2.3=1.189  v3.3=1.184.
J4_COST_CALIBRATION = float(os.environ.get("DPS_J4_CAL", "1.22"))

# J6 (salinidad costera) ELIMINADO del problema multiobjetivo: sin zeta (interfaz
# SWI2) la salinidad solo es discriminable de forma gruesa (régimen salino vs
# fresco, con sesgo en escenarios frescos). El riesgo de intrusión queda capturado
# indirectamente por J1 (storage bajo) y J4 (costo de desal/camiones).

# ─── Población y áreas: valores base (escalable con multiplicador) ─────────
POP_GROWTH_RATES = (0.02, 0.05)              # 2% y 5% anual
AREA_MULTIPLIERS = (1.00, 0.85, 0.50)        # sin cambio / -15% / -50%

# ─── Climate (GCMs disponibles) ────────────────────────────────────────────
GCM_LIST = [
    ("MPI-ESM1-2-LR", "ssp585"),
    ("ACCESS-CM2",    "ssp585"),
    ("CanESM5",       "ssp585"),
    ("GFDL-ESM4",     "ssp585"),
]

# ─── Optimizador (Platypus) ────────────────────────────────────────────────
OPTIMIZER_CONFIG = {
    "algorithm":    "NSGAII",        # alternativa: "EpsMOEA"
    "population":   100,
    "evaluations":  20_000,           # exploración inicial; subir para producción
    "n_climate_scenarios": 5,         # escenarios climáticos por evaluación
    "seed":         42,
}

# ─── Sanity / logging ──────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
