# Bayesian Belief DPS

This repository contains the research code for the Bayesian Dynamic Policy Search (BayesDPS) experiments described in the accompanying paper. The project explores reservoir expansion planning under climate uncertainty using dynamic decision rules trained with multi-objective evolutionary optimization. Both Bayesian and non-Bayesian (normDPS) formulations are supported.

## Repository structure

- `main_par.py`: Entry point that launches multi-seed optimizations for both BayesDPS and normDPS configurations and writes pickled result bundles to `results/`.
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

   The default settings evaluate both BayesDPS and normDPS policies across multiple random seeds using the EpsMOEA algorithm. Results are written to `results/results_bayes_<label>.dat` and `results/results_norm_<label>.dat`.

3. Use `src/plot_optimization.py` to visualize Pareto fronts or compare BayesDPS versus normDPS performance. The plotting script expects the pickled result files produced by the previous step.

## Customization

- **Optimization controls:** Adjust population size, generation count, and seed count in `OptimizationParameters` within `main_par.py`.
- **Policy structure:** Modify the number of hidden nodes, inputs, or outputs in `OptimizationParameters` to change the neural/RBF policy dimensionality.
- **Cost and demand settings:** Update reservoir capacities, cost coefficients, and demand targets in `src/Config.py` (and the test-specific `Config_test*.py` files) to explore alternative scenarios.
- **Experiment scope:** Set `option_type` in `main_par.py` to `"both"`, `"static"`, or `"flexible"` to constrain expansion strategies.

## Reproducing figures

The notebook and plotting helpers in `src/` illustrate indicator processing, Gaussian process regression, and hypervolume metrics. For paper figures, load the saved `.dat` result files and reuse the plotting utilities to regenerate the Pareto comparisons described above.

## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.
