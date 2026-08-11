# -*- coding: utf-8 -*-
"""
main_robust_weap.py — Robust DPS sobre un ensamble climate × demand.

AISLADO del baseline: subclasea PipeWEAP y reescribe simulation() con la métrica
robusta mean + λ·std por objetivo sobre los escenarios. NO toca main_par_weap.py
ni pipe_simulation_weap.py.

  DPS_CKPT=<ckpt> python weap_dps/main_robust_weap.py \
      --evaluations 4000 --population 100 --seed 42 \
      --n_climate 5 --lam 1.0 --output runs_weap/robust/pareto_v3_seed42.dat
"""
from __future__ import annotations
import argparse, logging, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from platypus import NSGAII
from weap_dps.config_weap import (
    OPTIMIZER_CONFIG, RESULTS_DIR, ZARR_TEMPLATE_PATH,
    SPIN_UP_YEARS, DECISION_YEARS, WARMUP_WEEKS, WEEKS_PER_YEAR, j4_calibration_factor,
    OBJ_OPT_IDX, OBJECTIVES_OPTIMIZED, OBJECTIVES_DIAGNOSTIC, OBJECTIVE_NAMES,
)
from weap_dps.pipe_simulation_weap import PipeWEAP
from weap_dps.pipe_problem_weap import PipeProblemWEAP
from weap_dps.action_translator import ACTION_NAMES_BINARY, ACTION_NAMES_QUANTITY
from weap_dps.cost_calculator import compute_objectives
from weap_dps.scenario_builder import build_scenarios

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ROBUST] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


class RobustPipeWEAP(PipeWEAP):
    """simulation() robusto: agrega los objetivos por mean + λ·std sobre escenarios."""

    def __init__(self, *a, lam: float = 1.0, **k):
        super().__init__(*a, **k)
        self.lam = lam
        # J1 y J6 se calculan pero NSGA no los optimiza. Se cachean por politica
        # para poder reportarlos sobre el frente final sin re-evaluarlo (serian
        # ~100 politicas x 15 escenarios = 45 min extra por semilla).
        self._diag = {}

    @staticmethod
    def _key(P) -> tuple:
        return tuple(np.round(np.asarray(P, dtype=float), 12).tolist())

    def all_objectives_for(self, P) -> np.ndarray | None:
        """Los 7 objetivos de una politica ya evaluada (None si no esta)."""
        return self._diag.get(self._key(P))

    def simulation(self, P: np.ndarray):
        policy_fn = self._build_policy_from_params(P)
        all_J = []
        for X_scen in self.scenarios:
            result = self.surrogate.rollout_with_policy(
                X_template=X_scen, policy_fn=policy_fn, n_years=DECISION_YEARS,
                action_col_idx=self.action_col_idx, spin_up_years=SPIN_UP_YEARS,
            )
            gw = self.surrogate.denormalize_y(result["gw"], kind="gw")
            surf = self.surrogate.denormalize_y(result["surface"], kind="surface")
            if self.waterfall is not None:
                surf = self.waterfall.apply(surf, result["X_used"])
            objs = compute_objectives(
                gw_denorm=gw, surf_denorm=surf,
                target_names_gw=self.target_names_gw, target_names_surf=self.target_names_surf,
                actions_history=result["actions_history"],
                action_names_order=ACTION_NAMES_BINARY + ACTION_NAMES_QUANTITY,
                decision_start_week=WARMUP_WEEKS + SPIN_UP_YEARS * WEEKS_PER_YEAR,
                ap_demand_m3s=self._ap_demand(X_scen),
                ap_town_order=self.ap_town_order,
            )
            all_J.append(list(objs.values()))
        A = np.array(all_J, float)                          # (n_scen, 7) convención WEAP
        # J4: factor según cuántas acciones enciende ESTA política (el sesgo del
        # surrogate crece con el nº de acciones; ver config_weap.j4_calibration_factor).
        n_act = 0
        ah = result.get("actions_history")
        if ah is not None and len(ah):
            ah = np.asarray(ah)
            n_act = int((ah[:, :len(ACTION_NAMES_BINARY)] > 0.5).any(axis=0).sum())
        cal = j4_calibration_factor(n_act)
        # pasar a convención NSGA (J1,J3 a min) ANTES de mean/std, luego robustez
        M = np.column_stack([-A[:, 0], A[:, 1], -A[:, 2], A[:, 3] * cal,
                             A[:, 4], A[:, 5], A[:, 6]])    # J51, J52, J6
        robust = M.mean(0) + self.lam * M.std(0)            # mean + λ·std por objetivo
        # Los 7 quedan guardados; NSGA solo recibe los que discriminan.
        self._diag[self._key(P)] = robust.copy()
        return tuple(robust[OBJ_OPT_IDX].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluations", type=int, default=4000)
    p.add_argument("--population", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_climate", type=int, default=5)
    p.add_argument("--lam", type=float, default=1.0, help="aversión al riesgo (mean + λ·std)")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / f"robust_{int(time.time())}.dat")
    args = p.parse_args()

    np.random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pipe = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=args.lam)
    scen, labels = build_scenarios(pipe.surrogate, pipe.feature_names, pipe.X_template,
                                   n_climate=args.n_climate)
    pipe.scenarios = scen
    logger.info("Ensamble robusto: %d escenarios (clima×demanda)  λ=%.2f", len(scen), args.lam)
    for lb in labels:
        logger.info("   - %s", lb)

    problem = PipeProblemWEAP(pipe)
    algo = NSGAII(problem, population_size=args.population)
    t0 = time.time(); algo.run(args.evaluations); el = time.time() - t0
    logger.info("Listo en %.1f min  | frente=%d", el / 60, len(algo.result))

    n_hit = sum(pipe.all_objectives_for(s.variables) is not None for s in algo.result)
    logger.info("Diagnostico (J1, J6) recuperado para %d/%d politicas del frente",
                n_hit, len(algo.result))

    with open(args.output, "wb") as f:
        pickle.dump({"result": [(list(s.variables), list(s.objectives)) for s in algo.result],
                     # los 7 objetivos por politica: los 5 optimizados mas J1 y J6
                     "all_objectives": [
                         (lambda v: None if v is None else list(v))(
                             pipe.all_objectives_for(s.variables))
                         for s in algo.result],
                     "objective_names": OBJECTIVE_NAMES,
                     "objectives_optimized": OBJECTIVES_OPTIMIZED,
                     "objectives_diagnostic": OBJECTIVES_DIAGNOSTIC,
                     "scenarios": labels, "lam": args.lam,
                     "config": vars(args), "elapsed": el}, f)
    logger.info("Guardado: %s", args.output)


if __name__ == "__main__":
    main()
