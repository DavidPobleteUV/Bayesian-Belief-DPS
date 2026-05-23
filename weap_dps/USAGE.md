# Manual de uso — weap_dps

Guía paso a paso para usar el bridge WEAP-HydroMLP ↔ Standard DPS en el
caso Quilimari.

---

## Índice

1. [Prerequisitos](#1-prerequisitos)
2. [Setup inicial](#2-setup-inicial)
3. [Paso 1 — Extracción de datos](#3-paso-1--extracción-de-datos)
4. [Paso 2 — Sanity check](#4-paso-2--sanity-check)
5. [Paso 3 — Optimización exploratoria](#5-paso-3--optimización-exploratoria)
6. [Paso 4 — Optimización de producción](#6-paso-4--optimización-de-producción)
7. [Paso 5 — Análisis del frente de Pareto](#7-paso-5--análisis-del-frente-de-pareto)
8. [Ajuste de parámetros](#8-ajuste-de-parámetros)
9. [Troubleshooting](#9-troubleshooting)
10. [Cómo extender el bridge](#10-cómo-extender-el-bridge)
11. [Ciclo iterativo MLP ↔ WEAP (active learning)](#11-ciclo-iterativo-mlp--weap-active-learning)

---

## 1. Prerequisitos

### Repositorios

| Repo | Rol | Ubicación esperada |
|---|---|---|
| **`WEAP_HydroMLP_RecursiveGW`** | Modelo MLP entrenado | `C:\Users\David\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW` |
| **`Bayesian-Belief-DPS`** | Optimizador + bridge | `C:\Users\David\Documents\GitHub_DPL\Bayesian-Belief-DPS` |

El bridge espera ambos repos **hermanos** dentro de `GitHub_DPL/`. Si están
en otra ubicación, ajusta `MODEL_REPO` en `config_weap.py`.

### Archivos en el repo del modelo

Antes de empezar, verifica que existan:

```powershell
cd C:\Users\David\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW
Test-Path runs\best_model.ckpt                         # debe ser True
Test-Path data\variables_mlp_weekly_filtered.csv       # debe ser True
Test-Path data\scalers_weap.npz                        # debe ser True
Test-Path data\transform_params_weap.npz               # debe ser True
Test-Path data\weap_weekly.zarr                        # debe ser True
```

Si falta alguno, antes de seguir corre en el repo del modelo:

```powershell
.\venv_HydroMLP\Scripts\Activate.ps1
python src/scripts/data_preprocessing/prepare_training.py --config configs/recursive_config.yaml
# (entrenar si aún no hay checkpoint)
python src/scripts/training/train_lightning.py --config configs/recursive_config.yaml
```

### Venv del DPS

Crea el venv del DPS si no existe:

```powershell
cd C:\Users\David\Documents\GitHub_DPL\Bayesian-Belief-DPS
python -m venv venv_DPS
.\venv_DPS\Scripts\Activate.ps1
pip install torch pytorch-lightning numpy pandas zarr scipy platypus-opt pyyaml
pip install -e ..\WEAP_HydroMLP_RecursiveGW
```

Verifica que el modelo se puede importar desde el venv del DPS:

```powershell
python -c "from rdm_mlp.models.lightning_module import WEAPHydroMLPLightning; print('OK')"
```

---

## 2. Setup inicial

Una sola vez:

```powershell
cd C:\Users\David\Documents\GitHub_DPL\Bayesian-Belief-DPS
.\venv_DPS\Scripts\Activate.ps1

# Crear carpetas necesarias (idempotente)
New-Item -ItemType Directory -Force -Path data_weap, data_weap\climate_base, runs_weap, tests | Out-Null
```

---

## 3. Paso 1 — Extracción de datos

Copia el checkpoint, scalers y manifest del repo modelo a `data_weap/`, y
extrae un template del input desde un run baseline del zarr merged.

```powershell
python weap_dps/extract_data.py
```

**Output esperado:**

```
HH:MM:SS [EXTRACT] INFO Repo modelo: ...\WEAP_HydroMLP_RecursiveGW
HH:MM:SS [EXTRACT] INFO Destino: ...\data_weap
HH:MM:SS [EXTRACT] INFO Copiado: best_model.ckpt → ...
HH:MM:SS [EXTRACT] INFO Copiado: variables_mlp_weekly_filtered.csv → ...
HH:MM:SS [EXTRACT] INFO Copiado: scalers_weap.npz → ...
HH:MM:SS [EXTRACT] INFO Copiado: transform_params_weap.npz → ...
HH:MM:SS [EXTRACT] INFO Archivos copiados: 4/4
HH:MM:SS [EXTRACT] INFO Template guardado: ...\X_template.npz  shape=(1872, 611)
HH:MM:SS [EXTRACT] INFO Done.
```

### Re-ejecución

`extract_data.py` es idempotente: si los archivos destino son más nuevos
que los del repo modelo, salta la copia. Útil cuando reentrenas el MLP
y solo quieres actualizar el checkpoint:

```powershell
# Forzar nueva copia (borra y reextrae)
Remove-Item data_weap\best_model.ckpt
python weap_dps/extract_data.py
```

### Usar otro baseline run

Por default usa `run_id=0` para construir el template. Si quieres usar
otro (ej. un escenario sin acciones con clima húmedo):

Edita la línea final de `extract_data.py`:

```python
build_X_template(baseline_run_id=42)   # tu run_id preferido
```

---

## 4. Paso 2 — Sanity check

Valida que el bridge completo funciona sin tocar el optimizador.

```powershell
python tests/test_mlp_surrogate.py
```

**Output esperado (4 tests pasan):**

```
HH:MM:SS [TEST] INFO ============================================================
HH:MM:SS [TEST] INFO Test 1: Load checkpoint
HH:MM:SS [TEST] INFO ============================================================
HH:MM:SS [TEST] INFO   n_x=611  n_gw=524  n_surface=142
HH:MM:SS [TEST] INFO   OK

HH:MM:SS [TEST] INFO ============================================================
HH:MM:SS [TEST] INFO Test 2: predict_horizon over template
HH:MM:SS [TEST] INFO ============================================================
HH:MM:SS [TEST] INFO   X template shape: (1872, 611)
HH:MM:SS [TEST] INFO   gw_pred shape:    (1872, 524)   (range ... → ...)
HH:MM:SS [TEST] INFO   surf_pred shape:  (1872, 142)   (range ... → ...)
HH:MM:SS [TEST] INFO   OK

HH:MM:SS [TEST] INFO ============================================================
HH:MM:SS [TEST] INFO Test 3: rollout_with_policy with dummy policy
HH:MM:SS [TEST] INFO ============================================================
HH:MM:SS [TEST] INFO   Action cols mapped:
HH:MM:SS [TEST] INFO     act_desalacion_costera → col ...
HH:MM:SS [TEST] INFO     ...
HH:MM:SS [TEST] INFO   gw output:      (1872, 524)
HH:MM:SS [TEST] INFO   surface output: (1872, 142)
HH:MM:SS [TEST] INFO   actions_history shape: (26, 6)
HH:MM:SS [TEST] INFO   OK

HH:MM:SS [TEST] INFO ============================================================
HH:MM:SS [TEST] INFO Test 4: compute_objectives
HH:MM:SS [TEST] INFO ============================================================
HH:MM:SS [TEST] INFO   Objectives:
HH:MM:SS [TEST] INFO     J1_gw_storage = ...
HH:MM:SS [TEST] INFO     J2_unmet_ap = ...
HH:MM:SS [TEST] INFO     J3_agri_value = ...
HH:MM:SS [TEST] INFO     J4_supply_cost = ...
HH:MM:SS [TEST] INFO     J5_weeks_failure = ...
HH:MM:SS [TEST] INFO   OK

HH:MM:SS [TEST] INFO ============================================================
HH:MM:SS [TEST] INFO All sanity tests passed.
HH:MM:SS [TEST] INFO ============================================================
```

### Qué validar manualmente

- Los valores `J1..J5` deben ser **finitos** (no NaN) para una corrida real.
  Si todos son NaN, probablemente `target_names_gw/surf` están vacíos en
  `pipe_simulation_weap.py`. Ver sección Troubleshooting.
- `actions_history` debe mostrar **ceros** durante los primeros 5 años
  (spin-up) y la política dummy después.

---

## 5. Paso 3 — Optimización exploratoria

Corre NSGA-II con pocas evaluaciones para verificar que el optimizador
conecta con todo end-to-end.

```powershell
python weap_dps/main_par_weap.py `
  --algorithm NSGAII `
  --evaluations 500 `
  --population 30 `
  --workers 1
```

**Tiempo aprox.:** 30–60 min en CPU según hardware.

**Output esperado:**

```
HH:MM:SS [DPS_WEAP] INFO Algorithm:    NSGAII
HH:MM:SS [DPS_WEAP] INFO Evaluations:  500
HH:MM:SS [DPS_WEAP] INFO Population:   30
HH:MM:SS [DPS_WEAP] INFO Variables:    ...
HH:MM:SS [DPS_WEAP] INFO Objectives:   5
HH:MM:SS [DPS_WEAP] INFO Done in 1234.5s
HH:MM:SS [DPS_WEAP] INFO Pareto front size: 28
HH:MM:SS [DPS_WEAP] INFO Saved Pareto: runs_weap/pareto_1715789012.dat
```

### Inspección rápida del frente

```python
import pickle
import numpy as np

with open("runs_weap/pareto_XXXXXXXXXX.dat", "rb") as f:
    data = pickle.load(f)

print(f"Frente size: {len(data['result'])}")
print(f"Config: {data['config']}")

# Mejores soluciones por cada objetivo
obj_names = ["J1_neg(storage)", "J2_unmet", "J3_neg(value)", "J4_cost", "J5_failure"]
objs = np.array([sol[1] for sol in data['result']])
for i, name in enumerate(obj_names):
    best_idx = np.argmin(objs[:, i])
    print(f"Best {name}: idx={best_idx}, value={objs[best_idx]}")
```

> Recordatorio: `J1` y `J3` están **negados** internamente para que NSGA
> siempre minimice. Para reportar, multiplicar por −1.

---

## 6. Paso 4 — Optimización de producción

Cuando el smoke test funcione, lanza la corrida real.

### Configuración recomendada (similar al paper Zhang et al.)

```powershell
python weap_dps/main_par_weap.py `
  --algorithm NSGAII `
  --evaluations 81000 `
  --population 100 `
  --workers 4 `
  --seed 42 `
  --n_scenarios 5 `
  --output runs_weap/pareto_main.dat
```

**Tiempo aprox.:** 12–36 horas en CPU 4 workers (depende del MLP en CPU).

### Paralelización

`--workers N` lanza N procesos en paralelo. Cada uno carga su propia copia
del checkpoint en RAM (~50 MB cada uno). Ajusta según RAM disponible:

| Workers | RAM extra | Recomendado para |
|---|---|---|
| 1 | 0 | Debug / smoke test |
| 4 | ~200 MB | Laptop / workstation |
| 8–16 | ~1 GB | Estación con muchos cores |

### Múltiples seeds

El paper Zhang corre 4 seeds en paralelo. Para replicar:

```powershell
@(42, 123, 456, 789) | ForEach-Object {
  Start-Process -NoNewWindow -FilePath python -ArgumentList "weap_dps/main_par_weap.py --seed $_ --output runs_weap/pareto_seed${_}.dat"
}
```

---

## 7. Paso 5 — Análisis del frente de Pareto

### Comparación entre seeds (consistencia)

```python
import pickle, numpy as np, matplotlib.pyplot as plt

seeds = [42, 123, 456, 789]
fronts = {}
for s in seeds:
    with open(f"runs_weap/pareto_seed{s}.dat", "rb") as f:
        fronts[s] = np.array([sol[1] for sol in pickle.load(f)['result']])

# Plot J2 vs J4 (unmet vs cost)
for s, F in fronts.items():
    plt.scatter(F[:, 3], F[:, 1], label=f"seed {s}", alpha=0.6)
plt.xlabel("J4 cost"); plt.ylabel("J2 unmet AP")
plt.legend(); plt.savefig("runs_weap/consistency.png")
```

### Hipervolumen

```python
from platypus import Hypervolume
# Implementar con reference point específico al case Quilimari
# Ver paper Zhang Apéndice B para detalle de cálculo
```

### Análisis de políticas no dominadas

Para cada solución del frente, los parámetros del policy NN están en
`sol[0]`. Para evaluarla individualmente:

```python
from weap_dps.pipe_simulation_weap import PipeWEAP
from weap_dps.config_weap import ZARR_TEMPLATE_PATH

pipe = PipeWEAP(template_path=ZARR_TEMPLATE_PATH)
J = pipe.simulation(np.array(sol[0]))
print(f"Objectives: {J}")
```

---

## 8. Ajuste de parámetros

### Cambiar tarifas / precios

Editar `config_weap.py`:

```python
PRECIO_PALTO_CLP_PER_KG = 1800.0         # subir precio palto
TARIFA_DESAL_COSTERA_CLP_PER_M3 = 1400.0 # subir costo desal
```

Re-ejecutar optimización.

### Cambiar horizonte

```python
# En config_weap.py
SPIN_UP_YEARS = 11        # 2014–2024 (default)
DECISION_YEARS = 36       # extender a 2060 (cuando se tenga el clima)
```

> **Cuidado:** el MLP fue entrenado hasta 2050. Extender más allá implica
> extrapolación del modelo — válido para sensibilidad, no para producción.

### Cambiar arquitectura del policy NN

```python
# En pipe_simulation_weap.py
PipeWEAP(template_path=..., policy_arch=(24, 6))  # más hidden
```

A mayor `M` (hidden), más expresiva la política pero más variables a
optimizar → corrida más larga.

### Cambiar umbral de falla

```python
J5_FAILURE_THRESHOLD_FRAC = 0.05   # más estricto (5% en vez de 10%)
```

### Generar runs nuevos para mejorar el MLP

Si el frente muestra políticas extremas que parecen extrapolación del MLP
(NaN/Inf, valores físicamente imposibles), conviene generar nuevos runs
WEAP en esa región del espacio y reentrenar:

```powershell
cd C:\Users\David\Documents\GitHub_DPL\WEAP_2_ZARR
python src/tools/generate_lhs_extreme_runs.py
# ... correr WEAP ...
python src/tools/merge_zarrs.py ...
# Volver al repo modelo y reentrenar
```

---

## 9. Troubleshooting

### `FileNotFoundError: Template no encontrado`

Corre primero `python weap_dps/extract_data.py`.

### `KeyError: 'act_desalacion_costera' no está en feature_names`

El manifest del MLP no incluye esa acción. Posibles causas:

- El MLP fue entrenado **antes** de agregar esa acción al catálogo.
- Confirmar en `data_weap/manifest_inputs.csv` qué acciones existen como
  inputs (`role=input` y `source=policy`).
- Si falta, eliminar de `ACTION_NAMES_BINARY/QUANTITY` en `config_weap.py`
  hasta retrenar.

### `Loss = NaN` o `J* = NaN`

Posibles causas:

- **`target_names_gw` y `target_names_surf` vacíos** en `PipeWEAP.__init__`.
  Solución: cargar del manifest:
  ```python
  import pandas as pd
  df = pd.read_csv(MANIFEST_PATH)
  targets = df[df['role'] == 'target']['column'].tolist()
  # Separar GW y surface según convención del modelo
  ```
- **Scalers o transform_params corruptos**. Re-extraer:
  ```powershell
  Remove-Item data_weap\scalers_weap.npz, data_weap\transform_params_weap.npz
  python weap_dps/extract_data.py
  ```

### `Out of memory` al cargar el ckpt en N workers

Reducir `--workers`. Cada worker tiene su copia del MLP (~50 MB) y del
template (~50 MB). Con 16 workers son ~1.6 GB extra.

### El optimizador corre rápido pero el frente es plano (todos iguales)

- Verificar que `policy_output_to_actions` está mapeando bien (correr el
  sanity test paso 3 e inspeccionar `actions_history`).
- Aumentar diversidad del `population_size`.
- Bajar el umbral binario de 0.5 a 0.3 para más activaciones.

### El frente es ruidoso (varía mucho entre seeds)

- Aumentar `--evaluations`.
- Aumentar `--n_scenarios` para promediar más escenarios climáticos.
- Cambiar a EpsMOEA con epsilons más finas: edita `algorithm` en
  `config_weap.py` a `"EpsMOEA"`.

### El MLP devuelve valores fuera de rango físico

El MLP puede extrapolar fuera del soporte de entrenamiento. Si las
predicciones salen disparadas (ej. GW storage negativo enorme), conviene:

- **Reentrenar** con más datos extremos (LHS extremo `IDs 774-859`).
- **Clippear** outputs en `cost_calculator.py` para acotar objetivos.

---

## 10. Cómo extender el bridge

### Agregar un 6º objetivo (ej. empleo agrícola)

1. En `cost_calculator.py`, agregar función `j6_employment(...)`:
   ```python
   def j6_employment(surf_denorm, target_names, areas_history, ...):
       # JH = área × productividad_JH_por_ha
       ...
   ```
2. En `compute_objectives()`, agregar `"J6_employment"` al dict.
3. En `pipe_simulation_weap.py`, agregar al tuple de retorno de `simulation()`.
4. En `pipe_problem_weap.py`, cambiar `super().__init__(n_vars, 5)` → `6`.
5. Re-correr.

### Agregar acciones nuevas (cuando se retrene el MLP)

1. Agregar a `ACTION_NAMES_BINARY` / `ACTION_NAMES_QUANTITY` en `config_weap.py`.
2. Actualizar `Q_BOUNDS` con los rangos físicos.
3. Verificar que el manifest del MLP retreaned tenga esas columnas.
4. Ajustar `policy_K` en `PipeWEAP.__init__` para que coincida con el nuevo
   número de acciones.

### Usar otro algoritmo de Platypus

```python
from platypus import IBEA, OMOPSO, SPEA2
# en main_par_weap.py
algo = IBEA(problem, population_size=args.population)
```

### Persistir checkpoints intermedios del optimizador

Modificar `main_par_weap.py` para guardar el estado del algoritmo cada N
generaciones — permite resumir corridas largas:

```python
from platypus import save_objectives
# cada 1000 evals
save_objectives([s.objectives for s in algo.result],
                f"runs_weap/checkpoint_{nfe}.csv")
```

---

## 11. Ciclo iterativo MLP ↔ WEAP (active learning)

Estrategia para refinar el MLP usando soluciones óptimas del DPS. Cada
iteración aumenta el dataset de entrenamiento con runs WEAP en la región
del espacio de políticas que el optimizador considera prometedora, y
reentrena el MLP para que su acuerdo con WEAP mejore.

### Esquema del ciclo

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
  │   │ Entrenar MLP │ →  │ Correr DPS   │ →  │ Pareto front │      │
  │   │  (WEAP_HMlp) │    │ (NSGA-II)    │    │  ≈ 50 sols   │      │
  │   └──────────────┘    └──────────────┘    └──────┬───────┘      │
  │         ▲                                         │              │
  │         │                                         ▼              │
  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
  │   │ Merge nuevos │ ←  │ Run WEAP     │ ←  │ Export 21    │      │
  │   │   al zarr    │    │  21 runs     │    │ runs (CSV +  │      │
  │   │              │    │ (~15 días)   │    │  schedules)  │      │
  │   └──────────────┘    └──────────────┘    └──────────────┘      │
  │         │                                                        │
  │         ▼                                                        │
  │   ┌──────────────┐                                               │
  │   │ Compare      │                                               │
  │   │ MLP vs WEAP  │                                               │
  │   │ → divergence │                                               │
  │   └──────┬───────┘                                               │
  │         │                                                        │
  │         ▼                                                        │
  │   ¿KGE > 0.7?  → STOP                                            │
  │   ¿KGE < 0.7?  → otra iteración                                  │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
```

### Estructura de archivos por iteración

```
Bayesian-Belief-DPS/data_weap/exports/
├── iter_01/
│   ├── Policies/                                  ← 21 CSVs (formato WEAP)
│   │   ├── policy_iter01_1000_extremo_J1_MPI-ESM1-2-LR.csv
│   │   ├── policy_iter01_1001_extremo_J1_ACCESS-CM2.csv
│   │   ├── ...
│   │   └── policy_iter01_1020_balanced_2_GFDL-ESM4.csv
│   ├── RunIDs_Q_pareto_iter01.csv                ← master CSV con metadata
│   ├── metadata.json                              ← policy params + objectives
│   ├── mlp_predictions.npz                        ← cache para compare (post-WEAP)
│   └── comparison/                                ← generado por compare_mlp_vs_weap.py
│       ├── divergence_per_run.csv
│       ├── divergence_summary.png
│       └── summary.json
├── iter_02/
│   └── ...
```

### Paso 11.1 — Exportar el Pareto a runs WEAP

Después de correr `main_par_weap.py` (Paso 4) y obtener `runs_weap/pareto_*.dat`:

```powershell
python weap_dps/pareto_to_runids.py `
    --pareto runs_weap/pareto_1715789012.dat `
    --iteration 1 `
    --start_id 1000
```

Esto crea:
- 21 CSVs de schedule en `data_weap/exports/iter_01/Policies/`
- `RunIDs_Q_pareto_iter01.csv` con IDs 1000–1020.
- `metadata.json` con policy params, objectives MLP y hash del ckpt.

**Selección de soluciones (7 por iteración):**

| # | Tipo | Por qué |
|---|---|---|
| 1 | Extremo `min J1` | Política maximiza GW storage |
| 2 | Extremo `min J2` | Política minimiza Unmet AP |
| 3 | Extremo `min J3` | Política maximiza valor agrícola |
| 4 | Extremo `min J4` | Política minimiza costo |
| 5 | Extremo `min J5` | Política minimiza semanas en falla |
| 6 | Balanceada (cluster centroide 1) | Compromiso región A del frente |
| 7 | Balanceada (cluster centroide 2) | Compromiso región B del frente |

Cada solución se evalúa bajo **3 climas** (MPI-ESM1-2-LR, ACCESS-CM2, GFDL-ESM4 — todos ssp585) → 7×3 = 21 runs.

### Paso 11.2 — Copiar a la carpeta WEAP

Manualmente (o con un script de copia que puedes agregar después):

```powershell
$src_iter = "C:\Users\David\Documents\GitHub_DPL\Bayesian-Belief-DPS\data_weap\exports\iter_01"
$weap_area = "C:\Users\David\Documents\WEAP Areas\Quilimari_WEAP_MODFLOW_RDM"
$weap_2_zarr = "C:\Users\David\Documents\GitHub_DPL\WEAP_2_ZARR"

# 1. Copiar CSVs de schedule a la carpeta WEAP
New-Item -ItemType Directory -Force -Path "$weap_area\Policies\iter_01" | Out-Null
Copy-Item -Path "$src_iter\Policies\*" -Destination "$weap_area\Policies\iter_01\" -Force

# 2. Copiar master CSV al WEAP_2_ZARR data/
Copy-Item -Path "$src_iter\RunIDs_Q_pareto_iter01.csv" `
          -Destination "$weap_2_zarr\data\RunIDs_Q_pareto_iter01.csv" -Force
```

### Paso 11.3 — Modificar `weap_runner.py` (una sola vez)

En el repo `WEAP_2_ZARR`, modificar `src/pipeline/weap_runner.py` para que
detecte la columna `policy_schedule_csv` en el row del RunIDs y use
expresiones `ReadFromFile()` en lugar de constantes para las acciones:

```python
# En set_run_inputs(), después de leer las acciones constantes:
policy_csv = row.get("policy_schedule_csv")
if isinstance(policy_csv, str) and policy_csv.strip():
    # Modo schedule: leer acciones desde CSV
    for action_name in ACTION_NAMES_BINARY:
        expr = f'ReadFromFile({policy_csv}, "{action_name}", , Average, , Interpolate)'
        WEAP.BranchVariable(f"\\Key\\Acciones\\{action_name}").Expression = expr
else:
    # Modo legacy: constantes 0/1
    ...
```

(El path exacto del branch en WEAP depende de cómo esté el modelo. Si tu
modelo WEAP tiene los `act_*` como Key Assumptions, usa
`\Key\Acciones\act_X`. Si están como variables de demand-supply links,
ajustar al path correcto.)

Luego actualizar `WEAP_2_ZARR/config/config.yaml`:

```yaml
weap:
  runids_lhs_files:
    - data/RunIDs_Q_lhs.csv
    - data/RunIDs_Q_lhs_extreme.csv
    - data/RunIDs_Q_pareto_iter01.csv    # <-- nuevo
```

### Paso 11.4 — Correr los 21 runs en WEAP

Dividido entre las 2 PCs (10–11 runs cada una, ~7 días por PC):

```powershell
# PC_oficina
$ids = 1000..1010
python src/pipeline/run_pipeline.py --config config/config.yaml --run_ids $ids --pc_name PC_oficina

# PC_Servidor
$ids = 1011..1020
python src/pipeline/run_pipeline.py --config config/config.yaml --run_ids $ids --pc_name PC_Servidor
```

### Paso 11.5 — Mergear nuevos runs al zarr

Cuando ambas PCs terminen:

```powershell
cd C:\Users\David\Documents\GitHub_DPL\WEAP_2_ZARR
python src/tools/merge_zarrs.py `
    --inputs results/training_data/merged_new/weap_weekly.zarr `
             results/training_data/PC_oficina/weap_weekly.zarr `
             results/training_data/PC_Servidor/weap_weekly.zarr `
    --output results/training_data/merged_iter01/weap_weekly.zarr
```

### Paso 11.6 — Comparar MLP vs WEAP

```powershell
cd C:\Users\David\Documents\GitHub_DPL\Bayesian-Belief-DPS

# Antes de comparar, cachear predicciones MLP para los mismos 21 runs
python -c @"
import numpy as np
from weap_dps.mlp_surrogate import MLPSurrogate
from weap_dps.pipe_simulation_weap import PipeWEAP
# Reconstruir cada policy y guardar predicciones MLP
# (script auxiliar; te puedo armar uno dedicado si lo necesitas)
"@

# Luego comparar
python weap_dps/compare_mlp_vs_weap.py `
    --iteration 1 `
    --weap_zarr ..\WEAP_2_ZARR\results\training_data\merged_iter01\weap_weekly.zarr
```

Genera `data_weap/exports/iter_01/comparison/` con:
- `divergence_per_run.csv` — KGE/NSE/RMSE/PBIAS por run
- `divergence_summary.png` — plot de paridad
- `summary.json` — estadísticas agregadas

**Criterio de convergencia recomendado**: si **KGE_median > 0.7** y
**|PBIAS_mean| < 15%** en al menos 6 de las 7 soluciones representativas,
considerar el ciclo convergido.

### Paso 11.7 — Reentrenar el MLP

Si la divergencia es alta, reentrenar el MLP con los nuevos runs:

```powershell
cd C:\Users\David\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW
.\venv_HydroMLP\Scripts\Activate.ps1

# Copiar el zarr actualizado
Copy-Item ..\WEAP_2_ZARR\results\training_data\merged_iter01\weap_weekly.zarr `
          data\weap_weekly.zarr -Recurse -Force

# Preparar splits y normalizar (con los nuevos runs en train)
python src\scripts\data_preprocessing\prepare_training.py --config configs\recursive_config.yaml --resplit

# Reentrenar
python src\scripts\training\train_lightning.py --config configs\recursive_config.yaml

# Backup del ckpt anterior y reemplazar
Copy-Item runs\best_model.ckpt runs\best_model_iter0.ckpt
# El nuevo best_model.ckpt queda automáticamente
```

Después, volver al **Paso 1** (`extract_data.py`) en el repo DPS para
actualizar la copia del ckpt y empezar la **iteración 2**.

### Cuántas iteraciones esperar

Típicamente:
- **Iteración 1**: KGE_median ~0.4–0.6 (MLP entrenado sin políticas DPS).
- **Iteración 2**: KGE_median ~0.6–0.7.
- **Iteración 3–4**: KGE_median ~0.75–0.85 (convergencia).

Si después de 5 iteraciones no converge, posibles causas:
- El espacio de políticas que explora el DPS es demasiado dispar al de
  entrenamiento. Ampliar el dataset original (LHS extremo, baseline diversa).
- La arquitectura del MLP es insuficiente. Subir `hidden_size` o agregar
  capas.
- Los target_names y feature_names tienen NaN sistemáticos.
  Revisar `data/variables_mlp_weekly_filtered.csv`.

---

## Referencias

- Zhang, M., Lickley, M., Zaniolo, M., Nellikkattil, A., Fletcher, S.M. (2026)
  *Bayesian Direct Policy Search for Adaptive Water Supply Planning with
  Endogenous Learning*. Water Resources Research (submitted).
- Poblete et al. (2025) *Robust Decision Making bajo cambio climático para
  la cuenca de Quilimari, Chile*. Proyecto Hidráulico, U. de Valparaíso.
- Hadka, D. (2015) *Platypus: A Free and Open Source Python Library for
  Multiobjective Optimization*.
