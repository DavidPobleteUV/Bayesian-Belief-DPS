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

# Zarr de entrenamiento del MODELO (lo usa scenario_builder para armar el
# ensamble climático: necesita los runs crudos, no solo el template).
# Se resuelve por orden de preferencia; override con DPS_TRAIN_ZARR.
def _resolve_train_zarr() -> Path:
    if os.environ.get("DPS_TRAIN_ZARR"):
        return Path(os.environ["DPS_TRAIN_ZARR"])
    # 1) subconjunto local (~30 MB): baseline + runs climáticos. Viaja con los
    #    artefactos del modelo, así el DPS corre SIN el repo del MLP ni el zarr
    #    completo de ~6 GB. Lo genera weap_dps/build_train_subset.py.
    local = DATA_DIR / "train_subset.zarr"
    if local.exists():
        return local
    # 2) dataset completo, si está disponible en el repo del modelo
    for rel in ("data/_v3_900_clean/weap_weekly_merged.zarr",   # iter1 (900 runs, limpio)
                "data/_v3_900/weap_weekly_merged.zarr",         # iter0 (900 runs)
                "data/weap_weekly.zarr"):                       # layout antiguo (773)
        p = MODEL_REPO / rel
        if p.exists():
            return p
    # Nada encontrado: se devuelve el path histórico igual, pero avisando. Sin
    # esto el fallo aparece más tarde como un traceback de zarr al abrir un
    # directorio inexistente, que no dice qué falta ni cómo arreglarlo.
    import warnings
    warnings.warn(
        f"No se encontró ningún zarr de entrenamiento.\n"
        f"  Buscado:\n"
        f"    1) $DPS_TRAIN_ZARR (no seteado)\n"
        f"    2) {local}\n"
        f"    3) {MODEL_REPO}/data/...\n"
        f"  Para correr el DPS basta el subconjunto (~16 MB): cópialo desde la PC\n"
        f"  donde se entrenó el modelo a  data_weap/train_subset.zarr\n"
        f"  (se genera con: python weap_dps/build_train_subset.py)",
        RuntimeWarning, stacklevel=2)
    return MODEL_REPO / "data" / "weap_weekly.zarr"             # fallback histórico


TRAIN_ZARR_PATH = _resolve_train_zarr()

# ─── Horizonte temporal ────────────────────────────────────────────────────
# MLP iter0_900 (900 runs): 2392 weeks (2014-04-02 → 2060-03, calendario WEAP).
# El modelo ANTERIOR era de 1872 weeks (→2050-03); si vuelves a ese checkpoint,
# hay que revertir este valor o el surrogate fallará por shape.
TOTAL_WEEKS_MLP = 2392

# Timeline real (week0 = 2014-04-02, año hidrológico inicia 2-abril):
#   warmup   : wk   0..103   = 2014-04 .. 2016-04  (2 años, recursión MLP)
#   spin-up  : wk 104..675   = 2016-04 .. 2027-04  (11 años, datos hist.)
#   decisión : wk 676..2391  = 2027-04 .. 2060-03  (33 años-agua disponibles)
# OJO: las decisiones DPS arrancan en 2027-04 (NO 2025). decision_start_week
#   = WARMUP + SPIN_UP*52 = 104 + 572 = 676 -> 2027-04-02 (verificado vs time[].
#   Por eso start_year del export = 2027 y BASE_YEAR = 2027.
SPIN_UP_YEARS = 11        # 2016-04 .. 2027-04
# 33 = TODO el horizonte del MLP nuevo: (2392 - 104 - 11*52)/52 = 1716/52 = 33
# años-agua de decisión (2027-04 .. 2060-03). Con el modelo viejo (1872 wk) solo
# cabían 23. Si cambias este valor, mueve ANALYSIS_HORIZON_Y igual para que J4
# anualice exactamente el período simulado.
DECISION_YEARS = 33
WARMUP_WEEKS = 104        # primeros 2 años, fixed por el MLP
WEEKS_PER_YEAR = 52

# Cuántas semanas usar en total = warmup + spin-up + decision
N_WEEKS_HORIZON = min(TOTAL_WEEKS_MLP, WARMUP_WEEKS + (SPIN_UP_YEARS + DECISION_YEARS) * WEEKS_PER_YEAR)

# ─── Espacio de acción ─────────────────────────────────────────────────────
# Las 5 acciones son BINARIAS PURAS (on/off). En los datos de entrenamiento del
# MLP cada acción aparece SOLO con un único valor de cantidad (q canónico), así
# que NO hay decisión continua: al activarse una binaria se inyecta su q fijo.
# Tratarlas como continuas haría que el MLP extrapole a cantidades nunca vistas.
# Set vigente (K=4): las dos acciones de prorrateo fueron ELIMINADAS del catálogo
# WEAP (no generaban mejoras) y se incorporó "acuerdo". Coincide con los 4 pares
# act_*/q_* que mergea el pipeline de WEAP_2_ZARR en el zarr de entrenamiento.
ACTION_NAMES_BINARY = [
    "act_desalacion_costera",
    "act_desalacion_completa",
    "act_nuevo_pozo_a_5km",
    "act_acuerdo",
]

# q canónico (valor único observado en el zarr de entrenamiento) que se inyecta
# cuando la binaria correspondiente está activa.
CANONICAL_Q = {
    "q_desalacion_costera":  0.1,    # planta pequeña
    "q_desalacion_completa": 0.3,    # planta grande
    "q_nuevo_pozo_a_5km":    0.12,
    "q_acuerdo":             0.025,  # reasignación agro→AP (sin obra)
}

# Nombres de las columnas q en el MISMO orden que ACTION_NAMES_BINARY.
ACTION_NAMES_QUANTITY = [
    "q_desalacion_costera",
    "q_desalacion_completa",
    "q_nuevo_pozo_a_5km",
    "q_acuerdo",
]

# Acciones de INFRAESTRUCTURA: su construcción es irreversible (built = cummax).
# El acuerdo NO está aquí: es administrativo, reversible y sin costo hundido.
ACTION_NAMES_INFRA = [
    "act_desalacion_costera",
    "act_desalacion_completa",
    "act_nuevo_pozo_a_5km",
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
# Debe coincidir con los años de decisión que SIMULA el rollout:
#   (N_WEEKS_HORIZON - WARMUP_WEEKS - SPIN_UP_YEARS*52) / 52
# Con el MLP de 2392 wk son 33 (antes 23, truncados por el horizonte de 1872).
# Si cambias DECISION_YEARS, cambia esto también o J4 anualizará un período
# distinto al simulado.
ANALYSIS_HORIZON_Y  = 33            # años-agua de análisis (2027-2060)
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
    "act_acuerdo": {
        "capex_clp":                0.0,      # administrativo, sin obra: solo OPEX
        "construction_lead_years":  0,
        "lifetime_years":           25,
    },
}

# ─── Umbrales / criterios de objetivos ─────────────────────────────────────
J5_FAILURE_THRESHOLD_FRAC = 0.10   # semana en falla si Unmet/Demand > 10%

# J5 se separó en dos porque medían cosas distintas y ninguna discriminaba:
#   J51 = promedio ENTRE PUEBLOS de sus semanas en falla (cronicidad)
#   J52 = peor año, deficit/demanda de la cuenca (severidad del extremo)
# La version anterior sumaba el deficit de todos los pueblos y lo comparaba
# con un umbral ABSOLUTO de 100 m3/semana = 0.27% de la demanda: bastaba que
# UN pueblo fallara un poco para marcar la semana, y el conteo se saturaba.
J51_THRESHOLD_FRAC = 0.10          # pueblo en falla esa semana si unmet/dem > 10%

# ─── Estado que observa la política del DPS ────────────────────────────────
# Fuente unica de verdad para N. Antes el 4 estaba repetido en tres archivos
# (pipe_problem, pipe_simulation) y desincronizarlos rompe el tamaño de P.
#
# NO incluye el calendario a proposito. Con year_idx como entrada, NSGA-II
# puede codificar cronogramas de lazo ABIERTO ("construir en el año 5") en vez
# de reglas de lazo cerrado ("construir si el acuífero baja de X"), que es
# justamente lo que un Direct Policy Search busca demostrar.
POLICY_STATE_FEATURES = [
    "gw_storage_avg",        # nivel medio del acuífero, ultimas 52 semanas
    "gw_trend",              # tendencia intra-anual
    "ap_unmet_frac",         # J2: deficit AP / demanda AP del ultimo año
    "truck_frac",            # J4: fraccion del suministro que viene en camiones
    "agr_unmet_idx",         # J3: deficit agricola (indice normalizado)
    "z_coastal",             # J6: cota de la interfaz salina en los pozos Q09
    "built_desalacion_costera",    # que obras YA existen (irreversibles)
    "built_desalacion_completa",
    "built_nuevo_pozo_a_5km",
]
N_STATE_FEATURES = len(POLICY_STATE_FEATURES)

# ─── Calibración de J4 (costo) ─────────────────────────────────────────────
# factor = E[costo_obs] / E[costo_pred] sobre las fuentes de RESPALDO (aducción,
# pozo costero, desal, acuerdo, camiones). Los pozos propios se excluyen: su
# costo es eléctrico y se cancela en el cociente.
#
# El surrogate SUBESTIMA el volumen de respaldo, y el sesgo NO es constante:
# crece con el número de acciones activas (medido en los 113 runs de test del
# modelo iter1_clean_h128):
#
#     acciones activas   n     factor
#            0          14      1.168
#            1          56      1.429
#            2          23      1.535
#           3+          20      2.067
#
# Por eso un ESCALAR único no es neutral al ranking: corregiría bien las
# políticas sin acciones y dejaría subestimadas las de muchas acciones (las
# caras), sesgando al optimizador hacia políticas agresivas. La calibración
# condicional quita ese sesgo diferencial (error mediano de J4: 32% sin
# calibrar -> 20% con escalar -> 16% condicional).
#
# El ORDEN sí lo preserva bien el surrogate: Spearman(costo_obs, costo_pred)
# = 0.92 sobre los 113 runs.
J4_CAL_BY_NACTIONS = {0: 1.168, 1: 1.429, 2: 1.535, 3: 2.067}   # 3 = "3 o más"

# Compat / override manual: si DPS_J4_CAL está seteado se usa ESE escalar para
# todos los casos (ignora la tabla). Útil para reproducir corridas antiguas.
J4_COST_CALIBRATION = float(os.environ["DPS_J4_CAL"]) if os.environ.get("DPS_J4_CAL") else None


def j4_calibration_factor(n_actions: int) -> float:
    """Factor de calibración de J4 según cuántas acciones están activas.

    `n_actions` = número de acciones ON en la política (0..4). Se agrupa 3+
    porque con 4 acciones hay pocos runs de test para estimar por separado.
    """
    if J4_COST_CALIBRATION is not None:      # override por entorno
        return J4_COST_CALIBRATION
    return J4_CAL_BY_NACTIONS[min(int(n_actions), 3)]

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
