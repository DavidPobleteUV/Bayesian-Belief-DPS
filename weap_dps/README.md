# weap_dps — Bridge WEAP-HydroMLP ↔ Standard DPS (Quilimari)

Namespace que conecta el surrogate **WEAP-HydroMLP** del proyecto hermano
`WEAP_HydroMLP_RecursiveGW` con el framework **Standard Direct Policy
Search** del paper Bayesian DPS (Zhang et al., 2026) para optimización
multiobjetivo de la cuenca de Quilimari (Chile).

> **Filosofía:** este namespace vive en paralelo a `src/` (intacto del paper
> original) para no mezclar código de referencia con código del case study.
> El `src/` queda como guía/comparación; toda la lógica del bridge está aquí.

---

## Índice

1. [Qué hace el bridge](#1-qué-hace-el-bridge)
2. [Arquitectura](#2-arquitectura)
3. [Estructura de archivos](#3-estructura-de-archivos)
4. [Dependencias](#4-dependencias)
5. [Quick start](#5-quick-start)
6. [Documentación detallada](#6-documentación-detallada)

---

## 1. Qué hace el bridge

Reemplaza el modelo de reservorio simple del Standard DPS por **WEAP-HydroMLP**,
un MLP recursivo de dos cabezas que predice 666 variables hidrológicas
(524 GW + 142 superficiales) a paso semanal, entrenado sobre 510 corridas
WEAP/MODFLOW del caso Quilimari.

La optimización (NSGA-II o EpsMOEA, vía **Platypus**) busca políticas de
adaptación que minimizan/maximizan 5 objetivos:

| Objetivo | Dirección | Descripción |
|---|---|---|
| **J1** GW storage | maximizar | mínimo de la suma de almacenamiento de los 9 SHACs |
| **J2** Unmet AP | minimizar | demanda no atendida total de Agua Potable (m³) |
| **J3** Valor agrícola | maximizar | producción de palto × precio (CLP) |
| **J4** Costo de abastecimiento | minimizar | tarifa × volumen por fuente (CLP) |
| **J5** Semanas en falla | minimizar | conteo de semanas con Unmet/Demand > 10% |

La política se modela como una red neuronal pequeña (Opción B, **adaptive**):
cada año la NN decide las 3 acciones binarias + 3 cantidades continuas en
base al estado hidrológico del año anterior.

---

## 2. Arquitectura

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Bayesian-Belief-DPS/                            │
│                                                                       │
│   src/                            ←  intacto (paper Zhang et al.)     │
│   ├── pipe_simulation.py          (referencia)                        │
│   ├── pipe_problem.py             (referencia)                        │
│   └── ...                                                             │
│                                                                       │
│   weap_dps/                       ←  bridge case Quilimari            │
│   │                                                                   │
│   │   ┌─────────────────┐                                             │
│   │   │ Platypus NSGA-II│ ────► optimiza policy params                │
│   │   └────────┬────────┘                                             │
│   │            │                                                      │
│   │            ▼                                                      │
│   │   ┌─────────────────┐                                             │
│   │   │ PipeProblemWEAP │  pipe_problem_weap.py                       │
│   │   └────────┬────────┘                                             │
│   │            │                                                      │
│   │            ▼                                                      │
│   │   ┌─────────────────┐                                             │
│   │   │   PipeWEAP      │  pipe_simulation_weap.py                    │
│   │   │  simulation(P)  │                                             │
│   │   └────────┬────────┘                                             │
│   │            │                                                      │
│   │            ├─► climate_sampler.py     (escenarios climáticos)     │
│   │            ├─► demand_builder.py      (pob × growth, áreas × mult)│
│   │            ├─► action_translator.py   (policy NN → 6 inputs MLP)  │
│   │            │                                                      │
│   │            ▼                                                      │
│   │   ┌─────────────────┐                                             │
│   │   │ MLPSurrogate    │  mlp_surrogate.py                           │
│   │   │ rollout_with_   │                                             │
│   │   │   policy()      │                                             │
│   │   └────────┬────────┘                                             │
│   │            │                                                      │
│   │            ▼                                                      │
│   │   ┌─────────────────┐                                             │
│   │   │ cost_calculator │  J1..J5                                     │
│   │   └─────────────────┘                                             │
│   │                                                                   │
│   data_weap/                     ←  artefactos extraídos              │
│   ├── best_model.ckpt             (copia del repo modelo)             │
│   ├── manifest_inputs.csv                                             │
│   ├── scalers_weap.npz                                                │
│   ├── transform_params_weap.npz                                       │
│   ├── X_template.npz              (input baseline shape (1872, 611))  │
│   └── climate_base/               (series GCM extraídas)              │
│                                                                       │
│   runs_weap/                      ←  frente de Pareto guardado        │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Estructura de archivos

```
weap_dps/
├── __init__.py
├── README.md                     ← este archivo
├── USAGE.md                      ← manual de uso detallado
│
├── config_weap.py                ← Constantes (horizonte, tarifas, GCMs)
├── extract_data.py               ← Copia artefactos del repo modelo
├── mlp_surrogate.py              ← Wrapper PyTorch del checkpoint
├── climate_sampler.py            ← Series weekly precip+temp por subcuenca
├── demand_builder.py             ← Crecimiento poblacional + áreas
├── action_translator.py          ← Policy NN output → input MLP
├── cost_calculator.py            ← J1..J5
├── pipe_simulation_weap.py       ← Loop anual adaptive
├── pipe_problem_weap.py          ← Wrapper Platypus
└── main_par_weap.py              ← Entry point optimization
```

---

## 4. Dependencias

Se asume el venv `Bayesian-Belief-DPS/venv_DPS/` con:

```
torch>=2.0
pytorch-lightning>=2.0
numpy>=1.24
pandas>=2.0
zarr>=2.16
scipy>=1.10
platypus-opt>=1.4
pyyaml>=6.0
rdm-mlp    (instalado editable desde ../WEAP_HydroMLP_RecursiveGW)
```

Setup completo:

```powershell
cd C:\Users\David\Documents\GitHub_DPL\Bayesian-Belief-DPS
python -m venv venv_DPS
.\venv_DPS\Scripts\Activate.ps1
pip install torch pytorch-lightning numpy pandas zarr scipy platypus-opt pyyaml
pip install -e ..\WEAP_HydroMLP_RecursiveGW
```

---

## 5. Quick start

Asumiendo venv activado y repo modelo accesible en `..\WEAP_HydroMLP_RecursiveGW`:

```powershell
# 1. Extraer ckpt + scalers + template
python weap_dps/extract_data.py

# 2. Sanity check (4 tests, ~30 segundos)
python tests/test_mlp_surrogate.py

# 3. Optimización mini (smoke test, ~5 min)
python weap_dps/main_par_weap.py --evaluations 200 --population 20

# 4. Optimización producción (varias horas)
python weap_dps/main_par_weap.py --evaluations 20000 --population 100 --workers 4
```

---

## 6. Documentación detallada

Para guía paso a paso, configuración, depuración y extensión, ver
[USAGE.md](USAGE.md).
