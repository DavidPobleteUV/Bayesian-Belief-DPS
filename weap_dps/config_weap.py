# -*- coding: utf-8 -*-
"""
config_weap.py — Constantes del case study Quilimari para Standard DPS.

Cualquier ajuste de hiperparámetro de la simulación, tarifas, áreas base o
horizonte debe pasar por aquí. NO hardcodear valores en otros módulos.
"""

from __future__ import annotations

from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data_weap"
CKPT_PATH    = DATA_DIR / "best_model.ckpt"
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

# Para esta prueba inicial: spin-up con datos observados/históricos hasta 2024,
# decisiones DPS desde 2025 hasta 2050 (26 años → 26 decisiones anuales).
SPIN_UP_YEARS = 11        # 2014–2024 (semanas 0..571)
DECISION_YEARS = 26       # 2025–2050 (semanas 572..1923, recortado a 1872)
WARMUP_WEEKS = 104        # primeras 2 años, fixed por el MLP
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

# Tarifa fija para Acuerdo (DemAGRO_SHAC_*_fict → town), CLP/m³
TARIFA_ACUERDO_CLP_PER_M3 = 3500.0

# ─── Parámetros para costo eléctrico de bombeo (pozos regulares) ───────────
# Aplica SOLO a transmission links desde APR_Q*_Fict_<town> (pozos baseline
# de cada utility). PozoCostero, Desal, Aduccion, Camiones, Acuerdo usan
# costos fijos del CSV.
PUMPING_RHO_KG_PER_M3   = 1000.0    # densidad del agua
PUMPING_G_M_PER_S2      = 9.81      # gravedad
PUMPING_EFFICIENCY      = 0.80      # eficiencia del sistema (80%)
PUMPING_EXTRA_LIFT_M    = 10.0      # carga adicional sobre depth_to_water
ENERGY_PRICE_CLP_PER_KWH = 250.0    # CLP/kWh
J_PER_KWH               = 3.6e6     # conversión J → kWh

# ─── Anualización de costos (NPV + EAC) ────────────────────────────────────
# Todos los flujos de caja (CAPEX y OPEX) se descuentan al año t=0 = BASE_YEAR.
# NSGA-II minimiza el NPV total. EAC se reporta solo para interpretación.
DISCOUNT_RATE       = 0.10          # 10% anual
BASE_YEAR           = 2025          # t = 0 (inicio decisiones DPS)
ANALYSIS_HORIZON_Y  = 23            # años de análisis (2025-2047)
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
