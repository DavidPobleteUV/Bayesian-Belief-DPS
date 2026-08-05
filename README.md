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
