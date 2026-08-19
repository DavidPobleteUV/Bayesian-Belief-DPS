# Bayesian Belief DPS

This repository contains the research code for the Bayesian-Belief Direct Policy Search (Bayes-Belief DPS) experiments described in the paper. The project explores long-term water infrastructure capacity expansion planning under climate uncertainty using adaptive policy trained with multi-objective evolutionary optimization. Both Bayesian and non-Bayesian (StandardPS) formulations are supported.

## Flujo entre los tres proyectos (cuenca Quilimari)

```text
  ┌────────────────────┐  zarr  ┌───────────────────────────┐  ckpt  ┌─────────────────────────┐
  │    WEAP_2_ZARR     │  RAW   │ WEAP_HydroMLP_RecursiveGW │ v2/v3  │   Bayesian-Belief-DPS   │
  │  simulación física │ ─────▶ │      surrogate (MLP)      │ ─────▶ │  optimización de política│
  │                    │        │                           │        │                         │
  │ WEAP–MODFLOW corre │        │ normaliza + entrena       │        │ DPS NSGA-II: policy NN  │
  │ los run_ids   →    │        │ v2/v3 (cascade) →         │        │ → 6 objetivos (J1..J6)  │
  │ weap_weekly.zarr   │        │ X_filtered / Y_filtered   │        │ Robust: clima × demanda │
  └─────────▲──────────┘        └───────────────────────────┘        └────────────┬────────────┘
            │                                                                      │
            │  run_XXXX.csv (acciones por valor de uso) +                          │ frente de
            │  RunIDs_Q_dps_proposals.csv                                          │ Pareto
            └──────────────────────────────────────────────────────────────────────┘
                 robust_pareto_to_rerun.py → loop-back: las propuestas se simulan en
                 WEAP y re-alimentan el surrogate (active learning)
```

Este repo es la **etapa de optimización** (derecha del diagrama, `weap_dps/`): consume el
checkpoint del surrogate y produce el frente de Pareto cuyas propuestas vuelven a WEAP.

## Estado (agosto 2026)

**Artefactos apuntando al emulador corregido.** `data_weap/` contiene el checkpoint
`iter1_fix2050` (época 57), sus scalers y sus parámetros de transformada; los anteriores quedaron
en `data_weap/_antiguo_iter1_sincorregir/`. Entre ellos estaba `train_subset.zarr`, que **tenía
precedencia sobre el zarr completo** en `_resolve_train_zarr()` y venía del dataset sin corregir:
el DPS lo habría usado en silencio. Los `x_mean` viejo y nuevo difieren hasta 0,075, de modo que
mezclar checkpoint y scalers de distintas iteraciones desalinea las entradas.

**Calibración residual de J4 plana.** Con el emulador corregido,
`J4_CAL_BY_NACTIONS = {0: 0.973, 1: 0.998, 2: 0.994, 3: 1.006}`. Antes crecía monótonamente con
el número de acciones —hasta 1,193 con tres o más— y esa deriva era el corte en 2050: a más
acciones activas, más semanas en que X las declaraba operando mientras Y no entregaba agua.
Aplicar la tabla antigua sobre este emulador penalizaría construir hasta en un 19 %.

**Cascada de despacho activa** (`DPS_WATERFALL=1` por omisión). El emulador reproduce el reparto
entre fuentes de WEAP, que responde a preferencias fijas y no a los costos; la cascada lo
sustituye por un despacho por **orden de mérito derivado de las tarifas**
(`waterfall_alloc.merit_order()`), las mismas con que `cost_calculator` factura. Dos correcciones
la hicieron utilizable:

- leía su registro de un zarr vacío, así que **nunca llegó a ejecutarse**: ahora usa
  `TRAIN_ZARR_PATH`;
- el **acuerdo de reasignación** estaba excluido y con sus enlaces en cero, lo que lo dejaba
  estrictamente dominado —costaba valor agrícola sin aportar agua urbana— y reducía el catálogo
  de cuatro acciones a tres. Ahora participa con su tope real de 25 L/s por localidad.

Es un **supuesto de modelación**: representa un operador que despacha por costo, mientras el
modelo de referencia despacha por prioridades fijas. Las políticas del frente que se re-simulen en
WEAP deben configurarse con las mismas tarifas y el mismo orden de preferencia.

**Rendimiento: un hilo de torch por proceso.** El rollout evalúa el MLP paso a paso con lote de
tamaño 1, y para matrices tan pequeñas coordinar hilos cuesta más de lo que rinde: 1 hilo son
381 µs/paso contra 454 µs con 6, y además usa 1 núcleo en vez de 5,9. Con 6 hilos cada semilla
acaparaba casi seis de los doce núcleos y las corridas en paralelo se peleaban por CPU. Medido en
corrida real: **39,5 s/evaluación con 1 núcleo**, contra 47,6 s con 5,9. Seis semillas × 4.000
evaluaciones caben en ~44 h sin contención. Configurable con `DPS_TORCH_THREADS`.

**Costos en MUSD.** El reporte muestra millones de USD a 980 CLP/USD (`USD_CLP_RATE`, definido una
sola vez y compartido con los gráficos). El cálculo y la optimización siguen en CLP: el tipo de
cambio solo afecta la presentación.

## Repository structure

- `main_par.py`: Entry point that launches multi-seed optimizations for both Bayesian-Belief DPS and StandardDPS configurations and writes pickled result bundles to `results/`.
- `src/Config.py`: Default experiment configuration (data paths, cost parameters, storage settings, and indicator normalization).
- `src/pipe_problem.py`: Platypus `Problem` definition that links candidate policy parameters to the simulation model.
- `src/pipe_simulation.py` / `src/pipesim_individual.py`: Core simulation logic for reservoir operation and infrastructure decisions.
- `src/policy.py`: Neural-network/RBF policy parameterization used during optimization.
- `src/plot_optimization.py`: Plotting utilities for Pareto fronts, hypervolume statistics, and policy behaviors.
- `data/`: Example climate and inflow indicator inputs used by the test configurations.

## Requirements

- Python 3.9+ is recommended.
- Key Python packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `numba`, `scikit-learn`, `statsmodels`, and `platypus-opt`.

Install dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn numba scikit-learn statsmodels platypus-opt
```

## Running an experiment

1. Ensure input data referenced in `src/Config.py` are available. By default the code expects indicator, runoff, and precipitation files under `data/Annual_GPR` (e.g., `test9_*` files). Adjust `Config.result_loc` and file names if you are using different datasets or storage paths.
2. Execute the driver script:

   ```bash
   python main_par.py
   ```

   The default settings evaluate both BayesDPS and StandardDPS policies across multiple random seeds using the EpsMOEA algorithm. Results are written to `results/results_bayes_<label>.dat` and `results/results_norm_<label>.dat`.

3. Use `src/plot_optimization.py` to visualize Pareto fronts or compare BayesDPS versus normDPS performance. The plotting script expects the pickled result files produced by the previous step.

## Customization

- **Optimization controls:** Adjust population size, generation count, and seed count in `OptimizationParameters` within `main_par.py`.
- **Policy structure:** Modify the number of hidden nodes, inputs, or outputs in `OptimizationParameters` to change the neural/RBF policy dimensionality.
- **Cost and demand settings:** Update reservoir capacities, cost coefficients, and demand targets in `src/Config.py` (and the test-specific `Config_test*.py` files) to explore alternative scenarios.
- **Experiment scope:** Set `option_type` in `main_par.py` to `"both"`, `"static"`, or `"flexible"` to constrain expansion strategies.



## Quilimari WEAP-HydroMLP DPS (`weap_dps/`)

Aplicación del DPS al caso Quilimari usando el surrogate WEAP-HydroMLP como modelo de
sistema (en vez del modelo analítico del paper). El optimizador NSGA-II entrena una policy
NN que decide **4 acciones** (desal costera, desal completa, nuevo pozo a 5 km, **acuerdo**)
sobre **6 objetivos** (J1 storage, J2 unmet, J3 agri, J4 cost, J5 failure weeks,
J6 salinidad costera).

> **Set de acciones (K=4).** Las dos acciones de *prorrateo* (SHAC/cuenca) fueron
> ELIMINADAS del catálogo WEAP por no generar mejoras, y se incorporó **acuerdo**
> (los nodos AP pueden extraer de los nodos subterráneos que abastecen demanda
> agrícola, a menor costo que el camión aljibe). Debe calzar con las columnas
> `act_*` de `../WEAP_2_ZARR/data/RunIDs_Q_full.csv`.
>
> **Horizonte.** Con el MLP `iter0_900` (900 runs, 2392 semanas: 2014-04 → 2060-03)
> el horizonte de decisión llega a **33 años (2027-2060)**; con el modelo anterior
> (1872 wk → 2050-03) se truncaba a 23. `TOTAL_WEEKS_MLP`, `DECISION_YEARS` y `ANALYSIS_HORIZON_Y`
> en `config_weap.py` deben moverse juntos al cambiar de checkpoint.

- `weap_dps/mlp_surrogate.py` — wrapper del checkpoint WEAP-HydroMLP (rollout año a año).
- `weap_dps/pipe_simulation_weap.py` / `pipe_problem_weap.py` — bridge simulación↔objetivos.
- `weap_dps/cost_calculator.py` — J1..J6 desde las salidas denormalizadas del MLP.
  J6 (salinidad costera) se **deriva del Z_value** predicho + geometría del pozo
  (`salinity_from_zvalue`), porque `WF_SalinityFactor` se sacó del entrenamiento
  por ser función determinista de ambos. Requiere `data_weap/reference/well_zbot.csv`
  (lo genera `weap_dps/build_well_zbot.py`).
- `weap_dps/main_par_weap.py` — DPS de **escenario único** (clima/demanda del run-0).
- Flags de entorno: `DPS_CKPT` (variante v2/v3), `DPS_WATERFALL` (.3, cascada well-anclada
  para J4), `DPS_J4_CAL` (calibración de costo por variante).

### Robust DPS (ensamble climate × demand)
- `weap_dps/scenario_builder.py` — arma el ensamble (N climas × corners de demanda)
  normalizando solo las columnas que cambian del template.
- `weap_dps/main_robust_weap.py` — `RobustPipeWEAP` (subclase) con métrica robusta
  **mean + λ·std** sobre los escenarios. Aislado del baseline.
- `run_robust.sh` — runner paralelo (1 proceso/core, `OMP=1`) sobre variantes × seeds.

### Exportar propuestas DPS a WEAP (formato re-corrida)
- `robust_pareto_to_rerun.py` — selecciona propuestas del frente robust (extremos +
  balanceadas + políticas con apagado), re-simula cada policy y escribe **`run_XXXX.csv`**
  (valor de uso por año) + master **`../WEAP_2_ZARR/data/RunIDs_Q_dps_proposals.csv`**
  (IDs 2000+), para correrlas en WEAP con `run_rerun_one.py` / `run_rerun_batch.py`.
  Respeta prendido/apagado; agrega variantes con sequía prolongada. Ejemplo:
  ```powershell
  $env:DPS_CKPT = "..\WEAP_HydroMLP_RecursiveGW\runs\iter07_v3_clean\best_model-epoch=011-val_loss=0.0638.ckpt"
  & "venv_DPS\Scripts\python.exe" robust_pareto_to_rerun.py --pareto runs_weap\robust\pareto_v3_seed42.dat
  ```

> El ciclo completo WEAP↔MLP↔DPS y el formato de acciones por valor de uso está en
> `../WEAP_2_ZARR/README.md` (§12) y `../WEAP_RERUN_PLAN.md`.

## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.
