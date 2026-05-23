# -*- coding: utf-8 -*-
"""
main_par_weap.py — Entry point de la optimización multiobjetivo Quilimari.

Carga el surrogate WEAP-HydroMLP, genera N escenarios climáticos, lanza
NSGA-II (o EpsMOEA) y guarda el frente de Pareto en runs_weap/.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

# Permite ejecutar como script: `python weap_dps/main_par_weap.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from platypus import NSGAII, EpsMOEA, ProcessPoolEvaluator

from weap_dps.config_weap import (
    OPTIMIZER_CONFIG, RESULTS_DIR, ZARR_TEMPLATE_PATH,
)
from weap_dps.pipe_simulation_weap import PipeWEAP
from weap_dps.pipe_problem_weap import PipeProblemWEAP

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [DPS_WEAP] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=["NSGAII", "EpsMOEA"],
                   default=OPTIMIZER_CONFIG["algorithm"])
    p.add_argument("--evaluations", type=int,
                   default=OPTIMIZER_CONFIG["evaluations"])
    p.add_argument("--population",  type=int,
                   default=OPTIMIZER_CONFIG["population"])
    p.add_argument("--seed",        type=int,
                   default=OPTIMIZER_CONFIG["seed"])
    p.add_argument("--workers",     type=int, default=1,
                   help="ProcessPool workers (1=serial)")
    p.add_argument("--n_scenarios", type=int,
                   default=OPTIMIZER_CONFIG["n_climate_scenarios"])
    p.add_argument("--output",      type=Path,
                   default=RESULTS_DIR / f"pareto_{int(time.time())}.dat")
    args = p.parse_args()

    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar template
    if not ZARR_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Template no encontrado: {ZARR_TEMPLATE_PATH}\n"
            f"Corre primero: python weap_dps/extract_data.py"
        )

    # Construir Pipe + Problem
    pipe = PipeWEAP(template_path=ZARR_TEMPLATE_PATH, scenarios=None)
    problem = PipeProblemWEAP(pipe)

    logger.info("Algorithm:    %s", args.algorithm)
    logger.info("Evaluations:  %d", args.evaluations)
    logger.info("Population:   %d", args.population)
    logger.info("Variables:    %d", problem.nvars)
    logger.info("Objectives:   %d", problem.nobjs)

    # Algoritmo
    if args.algorithm == "NSGAII":
        algo = NSGAII(problem, population_size=args.population)
    else:
        algo = EpsMOEA(problem, epsilons=[0.05]*5, population_size=args.population)

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolEvaluator(args.workers) as evaluator:
            algo.evaluator = evaluator
            algo.run(args.evaluations)
    else:
        algo.run(args.evaluations)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs", elapsed)
    logger.info("Pareto front size: %d", len(algo.result))

    # Guardar
    with open(args.output, "wb") as f:
        pickle.dump({
            "result": [(list(s.variables), list(s.objectives)) for s in algo.result],
            "config": vars(args),
            "elapsed": elapsed,
        }, f)
    logger.info("Saved Pareto: %s", args.output)


if __name__ == "__main__":
    main()
