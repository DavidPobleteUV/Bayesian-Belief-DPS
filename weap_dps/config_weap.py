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
# CKPT_PATH se resuelve mas abajo, junto con el resto de los artefactos.

# Cascada de despacho: J4 usa la asignación determinista anclada en el pozo
# propio (pozo nativo -> resto de fuentes por ORDEN DE MÉRITO) en vez del reparto
# nativo del emulador. El orden ya no está escrito a mano: lo deriva
# waterfall_alloc.merit_order() de UNIT_COST_BY_SOURCE, las mismas tarifas con
# que cost_calculator factura, de modo que despacho y costo no puedan divergir.
#
# ON por omisión desde iter1-fix2050. Antes estaba en 0 y además el allocator
# leía su registro de un zarr vacío, así que la cascada nunca llegó a correr:
# el reparto entre fuentes venía del emulador —es decir, de las preferencias de
# WEAP— y no respondía a los costos. Con la cascada activa, barrer tarifas sí
# mueve el reparto físico, que es lo que permite mapear vulnerabilidad.
#
# Supuesto de modelación asociado: se representa un operador que despacha por
# costo. Las políticas del frente que se re-simulen en WEAP deben configurarse
# con estas MISMAS tarifas y el mismo orden de preferencia entre fuentes; si no,
# emulador y modelo de referencia resuelven asignaciones distintas y la
# verificación mide esa diferencia en vez del error del emulador.
# DPS_WATERFALL=0 la apaga para reproducir corridas anteriores.
DPS_WATERFALL = os.environ.get("DPS_WATERFALL", "1") == "1"

# Correccion de balance: impone S + U = k*D, la identidad que WEAP cumple con
# CV 2.6% y el surrogate no (cierra 16% abajo y con 5x mas dispersion).
# Reduce el error de J4 de 25.2% a 7.6%. Ver weap_dps/balance_correction.py.
# ON por defecto; DPS_BALANCE=0 la apaga para reproducir corridas antiguas.
DPS_BALANCE = os.environ.get("DPS_BALANCE", "1") == "1"
# Artefactos del emulador. Los cinco van juntos: checkpoint, scalers, parámetros
# de transformada, manifest y template describen la MISMA iteración y mezclarlos
# desalinea columnas en silencio. Cada uno admite override por entorno para poder
# evaluar una iteración distinta sin tocar los archivos vigentes:
#
#   DPS_DATA_DIR=... (los cinco de una vez, si viven en la misma carpeta)
#   DPS_CKPT / DPS_SCALERS / DPS_TRANSFORM / DPS_MANIFEST / DPS_TEMPLATE
#
# El override individual manda sobre DPS_DATA_DIR.
_ART_DIR = Path(os.environ["DPS_DATA_DIR"]) if os.environ.get("DPS_DATA_DIR") else DATA_DIR


def _art(env: str, nombre: str) -> Path:
    return Path(os.environ[env]) if os.environ.get(env) else _ART_DIR / nombre


CKPT_PATH     = _art("DPS_CKPT", "best_model.ckpt")
MANIFEST_PATH = _art("DPS_MANIFEST", "manifest_inputs.csv")
SCALERS_PATH  = _art("DPS_SCALERS", "scalers_weap.npz")
TRANSFORM_PARAMS_PATH = _art("DPS_TRANSFORM", "transform_params_weap.npz")
CLIMATE_DIR   = DATA_DIR / "climate_base"
ZARR_TEMPLATE_PATH = _art("DPS_TEMPLATE", "X_template.npz")   # 1 run baseline

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
    # OJO: el orden importa. _v3_900_fix2050 es el único dataset cuyo X coincide
    # con lo que el modelo entregó: en los anteriores, las acciones figuran
    # activas después de 2050 mientras no entregan agua. Mezclar el checkpoint
    # actual con un zarr anterior también desalinea los scalers.
    for rel in ("data/_v3_900_fix2050/weap_weekly_merged.zarr",  # iter1 corregido
                "data/_v3_900_clean/weap_weekly_merged.zarr",    # iter1 sin corregir
                "data/_v3_900/weap_weekly_merged.zarr",          # iter0 (900 runs)
                "data/weap_weekly.zarr"):                        # layout antiguo (773)
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
    "Acuerdo":     1800.0,      # = 1500 al agricultor + 300 de transacción
}

# Desglose de la tarifa del acuerdo de reasignación. Solo la primera componente
# llega al predio; la segunda es costo de transacción y conducción. La distinción
# no cambia el costo del operador —J4 factura los 1800— pero sí el balance del
# agricultor, que es lo que determina si el instrumento consigue adhesión.
# Ver `balance_acuerdo.py` y §4.9 de la metodología.
ACUERDO_TARIFA_AGRICULTOR = 1500.0                  # CLP/m³ que recibe el predio
ACUERDO_COSTO_TRANSACCION = 300.0                   # CLP/m³ que no llega al predio
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
# Tipo de cambio para PRESENTACIÓN. Los objetivos se calculan y se optimizan en
# CLP; esta constante solo convierte para mostrar. Cambiarla no altera el frente,
# solo las unidades del reporte. Es la ÚNICA definición: no duplicarla en los
# módulos de gráficos.
USD_CLP_RATE        = 980.0         # CLP por USD

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

# ─── Ensamble climático del DPS ────────────────────────────────────────────
# Lista EXPLÍCITA de runs climáticos. Si está vacía, scenario_builder los elige
# solo, repartiendo por precipitación total acumulada.
#
# La selección automática tenía dos problemas:
#   1. Reparte sobre el ORDEN de precipitación, no sobre su distribución, y como
#      el extremo seco es un outlier el reparto queda torcido: los 5 climas
#      caían en los percentiles 0, 24, 47, 58 y 91 (dos casi pegados, y el 9%
#      más húmedo sin representar).
#   2. Mezcla proyecciones GCM con sequías SINTÉTICAS sin distinguirlas. El
#      ancla seca terminaba siendo una sequía de 30 años al 87% de severidad
#      (22.8 mm/año), que no es un clima sino un test de estrés — y como la
#      robustez es media + λ·std, ese escenario domina la desviación.
#
# Ensamble vigente: 9 climas, todos >= 110 mm/año.
#   5 series GCM puras cubriendo 127-248 mm/año (son las UNICAS disponibles:
#     5 modelos x 2 SSP, y solo 8 superan el piso)
#   4 con sequía impuesta de duración 10 años -> secuencias seco/húmedo con
#     recuperación, en vez de un desplazamiento permanente
#
# El run 555 (111 mm/año, racha de 21 años secos) queda FUERA a propósito: se
# reserva para el test de robustez posterior sobre el frente ya obtenido.
# Dos conjuntos, seleccionables con DPS_CLIMATE_SET.
#
#   "gcm"    (por omisión): las 8 proyecciones GCM SIN sequía impuesta. Optimizar
#            sobre futuros plausibles y dejar los severos para la verificación
#            separa dos preguntas que conviene no mezclar: si la política es buena
#            bajo lo esperable, y si además aguanta lo excepcional. Con sequías
#            dentro de la optimización, el frente se desplaza hacia políticas que
#            sobreconstruyen para un escenario que puede no ocurrir.
#   "mixto"  el conjunto anterior: 5 GCM + 4 sequías impuestas de 10 años.
#
# Precipitación media anual medida sobre 2014-2060 en las 6 subcuencas.
DPS_CLIMATE_RUNS_GCM = [
    0,    # 130.5 mm/año  CV 56%  NESM3/ssp585         - el más seco
    45,   # 158.4 mm/año  CV 39%  NESM3/ssp245         - seco y estable
    63,   # 184.5 mm/año  CV 47%  MPI-ESM1-2-LR/ssp245
    9,    # 184.8 mm/año  CV 58%  EC-Earth3-Veg/ssp585 - la mayor dispersión
    27,   # 201.0 mm/año  CV 41%  MPI-ESM1-2-LR/ssp585
    18,   # 202.0 mm/año  CV 53%  ACCESS-CM2/ssp585
    36,   # 216.3 mm/año  CV 47%  AWI-CM-1-1-MR/ssp585
    54,   # 234.1 mm/año  CV 46%  ACCESS-CM2/ssp245    - el más húmedo
]

DPS_CLIMATE_RUNS_MIXTO = [
    0,    # 130.5 mm/año  NESM3/ssp585
    45,   # 158.4 mm/año  NESM3/ssp245
    9,    # 184.8 mm/año  EC-Earth3-Veg/ssp585
    27,   # 201.0 mm/año  MPI-ESM1-2-LR/ssp585
    54,   # 234.1 mm/año  ACCESS-CM2/ssp245
    779,  # sequía sev 0.83 dur 10 desde 2040
    693,  # sequía sev 0.85 dur 10 desde 2040
    882,  # sequía sev 0.70 dur 10 desde 2035
    526,  # sequía sev 0.72 dur 10 desde 2025
]

_SET = os.environ.get("DPS_CLIMATE_SET", "gcm").lower()
if _SET not in ("gcm", "mixto"):
    raise SystemExit(f"DPS_CLIMATE_SET='{_SET}' no válido: usa 'gcm' o 'mixto'.")
DPS_CLIMATE_RUNS = DPS_CLIMATE_RUNS_GCM if _SET == "gcm" else DPS_CLIMATE_RUNS_MIXTO

# ─── Diseño de estados del mundo (SOW) ─────────────────────────────────────
# Tres incertidumbres, las mismas del diseño experimental de WEAP:
#   clima (DPS_CLIMATE_RUNS), crecimiento poblacional y cambio de uso de suelo.
#
# Antes eran 3 "corners" (HIGH/MID/LOW) que CONFUNDÍAN población y área: LOW
# bajaba el crecimiento a 2% Y reducía el área a la mitad al mismo tiempo, así
# que no se podía atribuir un efecto a un factor. Además el área solo tomaba dos
# valores (1.00 y 0.50) de los seis que tiene el diseño de WEAP.
#
# Un factorial completo 9 climas x 3 poblaciones x 4 áreas son 108 escenarios
# (226 h con 4000 evaluaciones). En su lugar se usa un diseño BALANCEADO: cada
# clima se combina con demandas distintas, y cada nivel de cada factor aparece
# el mismo número de veces. Con 27 escenarios —el mismo costo que antes— se
# cubren las 12 combinaciones población x área.
POP_LEVELS = {"2%": 0.02, "3%": 0.03, "5%": 0.05}       # los 3 del diseño WEAP
AREA_LEVELS = {"-50%": 0.50, "-15%": 0.85, "0%": 1.00, "+20%": 1.20}
DPS_N_SOW = 27      # 27 -> 56 h con 4000 evals | 36 -> 75 h (balance perfecto)

# ─── Estado que observa la política del DPS ────────────────────────────────
# Fuente unica de verdad para N. Antes el 4 estaba repetido en tres archivos
# (pipe_problem, pipe_simulation) y desincronizarlos rompe el tamaño de P.
#
# NO incluye el calendario a proposito. Con year_idx como entrada, NSGA-II
# puede codificar cronogramas de lazo ABIERTO ("construir en el año 5") en vez
# de reglas de lazo cerrado ("construir si el acuífero baja de X"), que es
# justamente lo que un Direct Policy Search busca demostrar.
# ─── Estado observado por la política ──────────────────────────────────────
# CRITERIO: toda variable del estado debe ser MEDIBLE en la cuenca al momento de
# decidir. Una política de lazo cerrado que dependa de una cantidad que solo
# existe dentro del modelo no es implementable por un operador real.
#
# El estado anterior fallaba ese criterio en tres de sus nueve variables:
#   gw_storage_avg  el almacenamiento agregado de 9 acuíferos es una salida de
#                   MODFLOW, no una medición
#   gw_trend        derivada del anterior
#   agr_unmet_idx   el déficit agrícola agregado no se monitorea sistemáticamente
# y una cuarta, z_coastal, aunque medible con sondas de conductividad, varía solo
# 0.2% sobre el frente: no aporta información para decidir.
#
# El estado nuevo replica lo que un operador SÍ observa: niveles en pozos de
# monitoreo, índices de precipitación acumulada, y sus propios registros de
# servicio (déficit y uso de camiones).
# Red de monitoreo: cinco pozos en cinco acuíferos distintos, de cabecera a
# costa. Se eligieron por KGE del emulador entre los que tienen métrica medida,
# para que la señal del estado sea confiable y no ruido del surrogate.
# Red de monitoreo: TRES pozos, uno por tramo de la cuenca. Se partió de cinco
# y el diagnóstico del smoke test mostró que no aportaban cinco señales: los
# pozos de Q06, Q07 y Q09 correlacionaban entre 0.86 y 0.95 entre sí, de modo
# que tres de las entradas de la política repetían información y solo agregaban
# dimensiones al espacio de búsqueda de NSGA-II. Se conservan los tres menos
# redundantes (correlación cruzada 0.66-0.74), uno por zona hidrogeológica.
POZOS_OBSERVACION = [
    "WF_DepthToWater_m__APR_Q01_Fict_JuntaTilama__Pozo1_15m",     # Q01 cabecera, KGE 0.957
    "WF_DepthToWater_m__APR_Q05_Fict_LosCondores__Pozo1_7m",      # Q05 medio,    KGE 0.815
    "WF_DepthToWater_m__DemAGRO_SHAC_Q09_fict__p_68_35_id17",     # Q09 costa,    KGE 0.973
]

# Dotación de subsistencia, litros por persona y día. Referencia: Howard et al.
# (2020), "Domestic water quantity, service level and health", 2.ª ed., OMS.
#
# NO se usa para acotar el déficit del estado de la política. Se intentó y la
# premisa resultó falsa: en el modelo la población es plana a lo largo del año
# mientras la demanda se triplica en verano, es decir, la población flotante está
# representada como un multiplicador de demanda sobre población residente
# constante. El consumo por habitante real se mantiene en 143 L/día, bajo este
# umbral, de modo que el corte clasificaba como discrecional el agua de
# subsistencia de los visitantes. Se conserva la constante porque el umbral sigue
# siendo el criterio correcto: lo que falta es separar residentes de flotantes.
DOTACION_SUBSISTENCIA_LPD = 200.0

POLICY_STATE_FEATURES = [
    # Índice de nivel subterráneo estandarizado, uno por pozo de monitoreo.
    # Adimensional y con la misma escala que el SPI, de modo que los tres
    # pozos y los dos índices climáticos son directamente comparables entre sí.
    "sgi_1",                 # Q01, cabecera
    "sgi_2",                 # Q05, zona media
    "sgi_3",                 # Q09, costa
    "spi_12",                # precipitación acumulada 52 sem, estandarizada
    "spi_24",                # idem 104 sem: señal climática plurianual
    # J2: déficit de agua potable del último año, relativo a la demanda total.
    # Ver `DOTACION_SUBSISTENCIA_LPD` sobre por qué no se acota a subsistencia.
    "ap_unmet_frac",
    "truck_frac",            # J4: fraccion del suministro que viene en camiones
    "built_desalacion_costera",    # que obras YA existen (irreversibles)
    "built_desalacion_completa",
    "built_nuevo_pozo_a_5km",
]
N_STATE_FEATURES = len(POLICY_STATE_FEATURES)

# Estado anterior, para reproducir corridas previas con DPS_STATE=legacy.
POLICY_STATE_FEATURES_LEGACY = [
    "gw_storage_avg", "gw_trend", "ap_unmet_frac", "truck_frac",
    "agr_unmet_idx", "z_coastal", "built_desalacion_costera",
    "built_desalacion_completa", "built_nuevo_pozo_a_5km",
]
if os.environ.get("DPS_STATE", "").lower() == "legacy":
    POLICY_STATE_FEATURES = POLICY_STATE_FEATURES_LEGACY
    N_STATE_FEATURES = len(POLICY_STATE_FEATURES)

# ─── Qué objetivos OPTIMIZA NSGA-II ────────────────────────────────────────
# Orden en que compute_objectives los devuelve (ver cost_calculator).
OBJECTIVE_NAMES = ["J1_gw_storage", "J2_unmet_ap", "J3_agri_value",
                   "J4_supply_cost", "J51_mean_town_fail",
                   "J52_worst_year_frac", "J6_coastal_salinity"]

# J1 y J6 salen del conjunto de optimización: sobre el frente varían 2.4% y
# 0.2% respectivamente, o sea no representan un trade-off. Mantenerlos solo
# sube la dimensión, y en dimensión alta casi todo queda no dominado por
# efecto geométrico (con 6 objetivos, 428 de 600 soluciones lo estaban).
# SE SIGUEN CALCULANDO: se reportan como diagnóstico sobre el frente final.
OBJECTIVES_OPTIMIZED = ["J2_unmet_ap", "J3_agri_value", "J4_supply_cost",
                        "J51_mean_town_fail", "J52_worst_year_frac"]
OBJECTIVES_DIAGNOSTIC = [o for o in OBJECTIVE_NAMES if o not in OBJECTIVES_OPTIMIZED]
OBJ_OPT_IDX = [OBJECTIVE_NAMES.index(o) for o in OBJECTIVES_OPTIMIZED]
N_OBJECTIVES = len(OBJ_OPT_IDX)

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
# = 0.83 sobre los 113 runs de test.
#
# Tabla RESIDUAL: se aplica DESPUÉS de la corrección de balance (DPS_BALANCE),
# que ya elimina la mayor parte del sesgo restaurando S + U = k·D. Aplicar la
# tabla antigua encima sería corregir dos veces.
#
# Medido sobre el emulador iter1_fix2050 (época 57), 113 runs de test:
#
#   sin corrección de balance        : sesgo 0.913 -> error 8.9%
#   con corrección de balance        : sesgo 1.005 -> error 4.4%
#
# La tabla residual quedó PLANA. En el emulador anterior el factor crecía
# monótonamente con el número de acciones —{0:1.008, 1:1.081, 2:1.063, 3:1.193},
# hasta 19% de subestimación con tres o más— y esa deriva era el corte en 2050:
# a más acciones activas, más semanas en que X decía "activa" mientras Y no
# entregaba agua. Corregido el dato, los cuatro factores caen dentro del ±2.7%
# de la unidad y la calibración empírica deja de hacer trabajo.
#
# Se conserva la tabla, con los valores nuevos, para no cambiar la interfaz;
# aplicar la ANTIGUA sobre este emulador introduciría un sesgo artificial de
# hasta +19% en el costo de las políticas con más obras, es decir, penalizaría
# construir — justo la decisión central del problema.
#
# Las tarifas usadas son las de UNIT_COST_BY_SOURCE; si esas cambian hay que
# recalcular (la mezcla de fuentes difiere entre observado y predicho, así que
# las tarifas NO se cancelan en el cociente).
# ── Recalibrado para el emulador de iter02 ────────────────────────────────
# Medido sobre los 123 runs de test de _v4_iter02, que cubren el ensamble
# completo (14 / 56 / 25 / 28 runs por grupo):
#
#     grupo   n   mediana   sd
#       0    14    1.039   0.048
#       1    56    1.021   0.077
#       2    25    1.032   0.083
#      3+    28    1.032   0.133
#
# La tabla sigue PLANA y cerca de la unidad: los cuatro grupos difieren en menos
# del 2% entre sí y ninguno se aparta más del 4% de 1.0, muy por debajo de su
# propia dispersión. La calibración empírica apenas hace trabajo —el error
# mediano de J4 pasa de 5.07% a 4.70%— y se conserva por interfaz.
#
# ADVERTENCIA sobre dónde se mide. Sobre las 49 corridas del FRENTE el factor da
# 1.13, no 1.03, porque allí el emulador subestima el costo un 12%. Ese sesgo es
# un efecto de estar FUERA DE DISTRIBUCIÓN, no una propiedad global del modelo:
# aplicar 1.13 a toda la búsqueda sobrecorregiría el ensamble base en ~10% y
# penalizaría artificialmente las carteras que el optimizador recorre para llegar
# al frente. La tabla se ajusta por tanto sobre el test, no sobre el frente.
J4_CAL_BY_NACTIONS = {0: 1.039, 1: 1.021, 2: 1.032, 3: 1.032}   # 3 = "3 o más"

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
