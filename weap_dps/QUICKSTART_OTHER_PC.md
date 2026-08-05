# QuickStart — Correr el workflow DPS↔MLP↔WEAP en una PC nueva

Manual completo desde cero. Asume Windows + PowerShell + Python 3.10+.

---

## Índice

1. [Resumen del workflow](#1-resumen-del-workflow)
2. [Pre-requisitos en la PC nueva](#2-pre-requisitos-en-la-pc-nueva)
3. [Setup automático con bootstrap](#3-setup-automático-con-bootstrap)
4. [Transferencia de artefactos pesados (no en GitHub)](#4-transferencia-de-artefactos-pesados-no-en-github)
5. [Ejecución del bridge — primer ciclo completo](#5-ejecución-del-bridge--primer-ciclo-completo)
6. [Iteraciones siguientes](#6-iteraciones-siguientes)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Resumen del workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ PC nueva                                                         │
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐           │
│  │ Bayesian-   │   │ WEAP_HydroMLP│   │ WEAP_2_ZARR │           │
│  │ Belief-DPS  │   │ _RecursiveGW │   │              │           │
│  │ (optimizer  │   │ (modelo MLP) │   │ (simulación │           │
│  │  bridge)    │   │              │   │  WEAP)       │           │
│  └─────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│        │                  │                   │                   │
│        └──────────┬───────┴───────────────────┘                   │
│                   │                                                │
│                   ▼                                                │
│           WEAP Area (Quilimari)                                    │
│           + GCM CSVs                                               │
│           + Policies/ (schedules generados)                        │
└─────────────────────────────────────────────────────────────────┘
```

3 repos GitHub + 1 carpeta de WEAP local + 3 venvs.

---

## 2. Pre-requisitos en la PC nueva

### Software a instalar manualmente

| Software | Versión | Notas |
|---|---|---|
| **Python** | 3.10+ (3.11 o 3.12 ok) | Asegurarse que `python` esté en PATH |
| **Git** | cualquiera reciente | `git --version` para verificar |
| **WEAP** | versión con licencia activa | Solo si vas a simular en WEAP en esta PC. Si solo corres el bridge DPS, no es necesario |
| **PowerShell** | 5.1 (default Windows) o 7 | El bootstrap usa sintaxis compatible con ambos |

### Verificación rápida

```powershell
python --version    # debe imprimir 3.10+
git --version       # debe responder
```

---

## 3. Setup automático con bootstrap

### 3.1 Clonar los 3 repos

```powershell
$REPOS_ROOT = "$HOME\Documents\GitHub_DPL"
New-Item -ItemType Directory -Force -Path $REPOS_ROOT | Out-Null
Set-Location $REPOS_ROOT

git clone https://github.com/DavidPobleteUV/WEAP_HydroMLP_RecursiveGW.git
git clone https://github.com/DavidPobleteUV/WEAP_2_ZARR.git
git clone https://github.com/DavidPobleteUV/Bayesian-Belief-DPS.git
# Nota: si el repo del DPS aún apunta al fork original (Mofan-coding),
# usar tu fork en su lugar. Si no tienes fork, contactar al admin.
```

### 3.2 Correr el bootstrap

```powershell
cd $REPOS_ROOT\Bayesian-Belief-DPS
.\bootstrap_new_pc.ps1
```

Esto:
- Crea `venv_HydroMLP` en `WEAP_HydroMLP_RecursiveGW` y `pip install -e .`
- Crea `venv_DPS` en `Bayesian-Belief-DPS` y `pip install -e ..\WEAP_HydroMLP_RecursiveGW`
- Crea `venv_WEAP2Zarr` en `WEAP_2_ZARR` (opcional, solo si vas a correr WEAP aquí)
- Verifica que `from rdm_mlp.models.lightning_module import WEAPHydroMLPLightning` funcione desde el venv del DPS
- Crea las carpetas `data_weap/`, `runs_weap/`, `tests/` en el repo DPS

Tiempo aprox: **5–8 minutos** (la mayor parte es pip install de torch).

---

## 4. Transferencia de artefactos pesados (no en GitHub)

Estos archivos NO están en GitHub porque son grandes. Hay que copiarlos
manualmente desde la PC original a la nueva (USB, OneDrive, SCP, etc.).

### Opción A (recomendada) — copiar solo los 4 artefactos que consume el DPS

Si en la PC nueva **solo vas a correr el DPS** (no re-entrenar el MLP), basta con
copiar la carpeta `Bayesian-Belief-DPS\data_weap\`. Son ~5 MB y ya vienen con
todo resuelto (índices, sub-set de columnas, template del horizonte correcto):

| Archivo (dentro de `data_weap\`) | Qué es | Tamaño |
|---|---|---|
| `best_model.ckpt` | checkpoint del MLP | ~5 MB |
| `scalers_weap.npz` | medias/desv. para denormalizar | ~30 KB |
| `transform_params_weap.npz` | log/arcsinh por variable | ~15 KB |
| `manifest_inputs.csv` | manifest FILTRADO del modelo | ~100 KB |
| `X_template.npz` | esqueleto de X + índices gw/surface | ~5 MB |

> ⚠️ Los cinco tienen que venir del **mismo modelo**. Mezclar un checkpoint nuevo
> con scalers viejos desnormaliza mal y **en silencio** (no lanza error).

**Verificación:**

```powershell
cd $REPOS_ROOT\Bayesian-Belief-DPS
foreach ($f in "best_model.ckpt","scalers_weap.npz","transform_params_weap.npz",
                "manifest_inputs.csv","X_template.npz") {
  "{0,-32} {1}" -f $f, (Test-Path "data_weap\$f")
}
# smoke test: carga el surrogate y valida dimensiones
python -c "import sys; sys.path.insert(0,'.'); from weap_dps.mlp_surrogate import MLPSurrogate; s=MLPSurrogate(); print(f'OK n_x={s.n_x} n_gw={s.n_gw} n_surface={s.n_surface}')"
```

Con el modelo `iter0_900` debe imprimir `OK n_x=519 n_gw=531 n_surface=126`.

### Opción B — copiar el repo del modelo completo (para re-entrenar)

| Archivo | Origen (PC vieja) | Destino (PC nueva) | Tamaño aprox |
|---|---|---|---|
| Zarr training | `WEAP_HydroMLP_RecursiveGW\data\_v3_900\weap_weekly_merged.zarr\` | mismo path | ~6 GB |
| Checkpoints | `WEAP_HydroMLP_RecursiveGW\runs\iter0_900\*.ckpt` | mismo path | ~30 MB |
| Scalers/transform/manifest | `WEAP_HydroMLP_RecursiveGW\data\_v3_900\*` | mismo path | ~150 KB |
| WEAP area (si vas a correr WEAP) | `C:\Users\<orig>\Documents\WEAP Areas\Quilimari_WEAP_MODFLOW_RDM\` | mismo path adaptado | ~varios GB |

Luego regenerar los artefactos del DPS:

```powershell
cd $REPOS_ROOT\Bayesian-Belief-DPS
python weap_dps\extract_data.py `
  --checkpoint "runs\iter0_900\best_model-epoch=042-val_loss=0.0533.ckpt" `
  --zarr     "data\_v3_900\weap_weekly_merged.zarr" `
  --manifest "data\_v3_900\variables_mlp_weekly_filtered.csv"
```

`--zarr` y `--manifest` son **obligatorios** si el modelo no está en el layout
antiguo (`data\weap_weekly.zarr`): de ellos salen el sub-set de columnas de X
(527→519) y los índices gw/surface que el surrogate necesita.

---

## 5. Ejecución del bridge — primer ciclo completo

### 5.1 Activar venv del DPS

```powershell
cd $HOME\Documents\GitHub_DPL\Bayesian-Belief-DPS
.\venv_DPS\Scripts\Activate.ps1
```

Verás `(venv_DPS)` en el prompt.

### 5.2 Extraer datos a `data_weap/`

```powershell
python weap_dps\extract_data.py
```

Output esperado:
```
[EXTRACT] Copiado: best_model.ckpt → ...
[EXTRACT] Copiado: variables_mlp_weekly_filtered.csv → ...
[EXTRACT] Copiado: scalers_weap.npz → ...
[EXTRACT] Copiado: transform_params_weap.npz → ...
[EXTRACT] Template guardado: ...\X_template.npz  shape=(1872, 611)
```

### 5.3 Sanity check (4 tests)

```powershell
python tests\test_mlp_surrogate.py
```

Tiempo: ~5 min. Debe terminar con:
```
[TEST] All sanity tests passed.
```

Si los 4 J's tienen valores numéricos (no NaN), el bridge está OK.

### 5.4 Optimización exploratoria (smoke test)

```powershell
python weap_dps\main_par_weap.py `
    --algorithm NSGAII `
    --evaluations 500 `
    --population 30 `
    --workers 1 `
    --output runs_weap\pareto_iter01_smoke.dat
```

Tiempo: **30–60 min** en CPU.

Verifica que el frente tenga al menos ~15–30 soluciones no-dominadas:

```powershell
python -c @"
import pickle
with open('runs_weap/pareto_iter01_smoke.dat', 'rb') as f:
    data = pickle.load(f)
print(f'Frente size: {len(data[\"result\"])}')
print(f'Elapsed: {data[\"elapsed\"]:.1f}s')
"@
```

### 5.5 Exportar 21 runs para WEAP

```powershell
python weap_dps\pareto_to_runids.py `
    --pareto runs_weap\pareto_iter01_smoke.dat `
    --iteration 1 `
    --start_id 1000
```

Esto crea:
- `data_weap\exports\iter_01\Policies\` con 21 CSVs en formato WEAP (`$Columns = Date,act_*,q_*,...`)
- `data_weap\exports\iter_01\RunIDs_Q_pareto_iter01.csv` — master con IDs 1000–1020
- `data_weap\exports\iter_01\metadata.json` — policy params, objectives, hash ckpt

### 5.6 Copiar schedules a la carpeta WEAP

```powershell
$src = "$PWD\data_weap\exports\iter_01"
$weap_area = "$HOME\Documents\WEAP Areas\Quilimari_WEAP_MODFLOW_RDM"

# Crear carpeta Policies\ si no existe
New-Item -ItemType Directory -Force -Path "$weap_area\Policies" | Out-Null

# Copiar los 21 CSVs (preservando solo el archivo, NO un subdirectorio iter_01)
Copy-Item -Path "$src\Policies\*.csv" -Destination "$weap_area\Policies\" -Force

# Copiar master CSV al data/ del WEAP_2_ZARR
Copy-Item -Path "$src\RunIDs_Q_pareto_iter01.csv" `
          -Destination "$HOME\Documents\GitHub_DPL\WEAP_2_ZARR\data\RunIDs_Q_pareto_iter01.csv" `
          -Force
```

### 5.7 Agregar el master CSV al config del pipeline WEAP

Editar `WEAP_2_ZARR\config\config.yaml` y agregar el archivo al listado:

```yaml
weap:
  runids_lhs_files:
    - data/RunIDs_Q_lhs.csv
    - data/RunIDs_Q_lhs_extreme.csv
    - data/RunIDs_Q_pareto_iter01.csv    # ← nuevo
```

### 5.8 Correr 1 run de prueba con `--set_only`

Antes de lanzar los 21 runs, valida que las expresiones WEAP queden bien:

```powershell
cd $HOME\Documents\GitHub_DPL\WEAP_2_ZARR
.\venv_WEAP2Zarr\Scripts\Activate.ps1

python src\pipeline\run_pipeline.py `
    --config config\config.yaml `
    --run_ids 1000 `
    --pc_name TestPC `
    --set_only
```

Esto sólo setea las expresiones en WEAP (sin Calculate). Abre WEAP a mano
y verifica visualmente que en el branch `Acciones2_full` las expresiones
aparezcan como:
```
If(ReadFromFile(Policies\policy_iter01_1000_..._MPI-ESM1-2-LR.csv, "act_desalacion_costera", , Average, , Interpolate) > 0.5, <Activacion>, <Desactivacion>)
```

Si las expresiones se ven bien y WEAP las acepta sin error, sigue con
los 21 runs reales.

### 5.9 Lanzar los 21 runs WEAP

Dividido entre las máquinas disponibles. En esta PC nueva (si tiene WEAP):

```powershell
$ids = 1000..1020   # los 21 IDs del iter_01
python src\pipeline\run_pipeline.py `
    --config config\config.yaml `
    --run_ids $ids `
    --pc_name $env:COMPUTERNAME 2>&1 | `
    Tee-Object -FilePath "results\pipeline\logs\pareto_iter01_$(Get-Date -Format yyyyMMdd_HHmm).log"
```

Tiempo: **~15 días** en una sola PC (~45 min × 21 runs). Si las divides
entre 2 PCs, ~7–8 días.

### 5.10 Mergear los nuevos runs al zarr

Cuando terminen los 21 runs:

```powershell
cd $HOME\Documents\GitHub_DPL\WEAP_2_ZARR
python src\tools\merge_zarrs.py `
    --inputs results\training_data\merged_new\weap_weekly.zarr `
             results\training_data\$env:COMPUTERNAME\weap_weekly.zarr `
    --output results\training_data\merged_iter01\weap_weekly.zarr
```

### 5.11 Comparar MLP vs WEAP

```powershell
cd $HOME\Documents\GitHub_DPL\Bayesian-Belief-DPS
.\venv_DPS\Scripts\Activate.ps1

python weap_dps\compare_mlp_vs_weap.py `
    --iteration 1 `
    --weap_zarr ..\WEAP_2_ZARR\results\training_data\merged_iter01\weap_weekly.zarr
```

Genera `data_weap\exports\iter_01\comparison\` con:
- `divergence_per_run.csv` — KGE/NSE/RMSE/PBIAS por run
- `divergence_summary.png` — plot de paridad
- `summary.json` — estadísticas agregadas

**Criterio de convergencia**: si `kge_median > 0.7` y `|pbias_mean| < 15%`,
el ciclo está casi convergido. Si no, ir al paso siguiente.

### 5.12 Reentrenar el MLP

```powershell
cd $HOME\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW
.\venv_HydroMLP\Scripts\Activate.ps1

# Actualizar el zarr de entrenamiento con los nuevos runs
Copy-Item ..\WEAP_2_ZARR\results\training_data\merged_iter01\weap_weekly.zarr `
          data\weap_weekly.zarr -Recurse -Force

# Re-preparar splits + normalización (incluye los nuevos runs)
python src\scripts\data_preprocessing\prepare_training.py `
    --config configs\recursive_config.yaml --resplit

# Reentrenar (varias horas)
python src\scripts\training\train_lightning.py `
    --config configs\recursive_config.yaml

# Backup del ckpt anterior
Copy-Item runs\best_model.ckpt runs\best_model_iter01.ckpt
```

El nuevo `best_model.ckpt` reemplaza al anterior automáticamente.

---

## 6. Iteraciones siguientes

Para la **iteración 2** y siguientes, repite los pasos 5.2–5.12 con:

```powershell
# 5.2: re-extract con el ckpt actualizado
python weap_dps\extract_data.py

# 5.4: nueva optimización (con MLP retreaned)
python weap_dps\main_par_weap.py --output runs_weap\pareto_iter02_smoke.dat

# 5.5: exportar con start_id incrementado
python weap_dps\pareto_to_runids.py `
    --pareto runs_weap\pareto_iter02_smoke.dat `
    --iteration 2 `
    --start_id 1021    # siguiente IDs después de iter_01
```

Y así sucesivamente. Esquema de IDs sugerido:

| Iteración | Rango IDs | Cantidad |
|---|---|---|
| iter_01 | 1000–1020 | 21 |
| iter_02 | 1021–1041 | 21 |
| iter_03 | 1042–1062 | 21 |
| ... | ... | ... |

---

## 7. Troubleshooting

### "ModuleNotFoundError: No module named 'rdm_mlp'"

El venv del DPS no tiene el paquete editable instalado. Solución:

```powershell
.\venv_DPS\Scripts\Activate.ps1
pip install -e ..\WEAP_HydroMLP_RecursiveGW
```

### "FileNotFoundError: best_model.ckpt"

Te falta transferir el ckpt desde la PC original (sección 4) o reentrenar
desde cero con `train_lightning.py`.

### El sanity test falla en Test 1 con error de carga

Verifica que el manifest, scalers y transform_params estén en `data/` del
repo modelo, no solo en `data_weap/` del DPS. `extract_data.py` los copia
DESDE el repo modelo HACIA `data_weap/`.

### El optimizador es muy lento

Causas comunes:
- `--workers 1` no paraleliza. Subir a 4–8.
- Cada evaluación toma ~5 min (rollout O(n²)). Cuando el sanity test pase,
  el siguiente paso será optimizar el rollout incremental.

### WEAP rechaza la expresión `If(ReadFromFile(...)>0.5, A, D)`

Posibles causas:
- El path del CSV no existe en la carpeta WEAP. Verifica que copiaste a
  `<WEAP_area>\Policies\` (sin subcarpeta `iter_01\`).
- La sintaxis exacta puede variar entre versiones de WEAP. Si rechaza,
  probar con paréntesis explícitos o sin el `> 0.5`.

### `git push` falla porque el remote es `Mofan-coding`

El repo Bayesian-Belief-DPS original es de otro autor. Tienes que:
1. Hacer fork en GitHub UI (botón "Fork" arriba a la derecha).
2. Cambiar el remote local:
   ```powershell
   git remote set-url origin https://github.com/<tu_usuario>/Bayesian-Belief-DPS.git
   ```
3. `git push origin main`.

### Differencias entre 2 PCs en los frentes de Pareto

Si corres la misma optimización con el mismo `--seed` en 2 PCs y obtienes
frentes distintos, posibles causas:
- Versión distinta de PyTorch/CUDA → mismo seed produce predicciones
  ligeramente distintas en el MLP.
- Una PC tiene un ckpt más nuevo (verifica `mlp_ckpt_hash` en `metadata.json`).

---

## Apéndice: archivos clave

| Archivo | Repo | Función |
|---|---|---|
| `bootstrap_new_pc.ps1` | Bayesian-Belief-DPS | Setup automático |
| `weap_dps/extract_data.py` | Bayesian-Belief-DPS | Copia ckpt + scalers + manifest |
| `weap_dps/main_par_weap.py` | Bayesian-Belief-DPS | Entry optimizer NSGA-II |
| `weap_dps/pareto_to_runids.py` | Bayesian-Belief-DPS | Exporta Pareto a CSVs WEAP |
| `weap_dps/compare_mlp_vs_weap.py` | Bayesian-Belief-DPS | Mide convergencia |
| `tests/test_mlp_surrogate.py` | Bayesian-Belief-DPS | Sanity check |
| `src/pipeline/weap_runner.py` | WEAP_2_ZARR | Lee `policy_schedule_csv` |
| `runs/best_model.ckpt` | WEAP_HydroMLP_RecursiveGW | Modelo entrenado |
| `data/weap_weekly.zarr` | WEAP_HydroMLP_RecursiveGW | Training data |
