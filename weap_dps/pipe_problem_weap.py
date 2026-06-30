# -*- coding: utf-8 -*-
"""
pipe_problem_weap.py — Wrapper Platypus para el problema multiobjetivo
de Quilimari (5 objetivos).
"""

from __future__ import annotations

import logging

import numpy as np
from platypus import Problem, Real

from weap_dps.pipe_simulation_weap import PipeWEAP

logger = logging.getLogger(__name__)


class PipeProblemWEAP(Problem):
    """
    5 objetivos. La cantidad de variables de decisión = tamaño del policy NN.
    Cada variable es continua en [-3, 3] (bounds amplios; la activación tanh
    hace de squashing).
    """

    def __init__(self, pipe: PipeWEAP,
                 var_lo: float = -3.0,
                 var_hi: float = 3.0):
        self.pipe = pipe
        n_vars = PipeWEAP.policy_param_size(
            M=pipe.policy_M, K=pipe.policy_K, N=4,
        )
        # 5 objetivos (J1..J5). J6 (salinidad costera) se ELIMINÓ: el surrogate solo
        # logra discriminación gruesa de régimen salino sin zeta (intrusión SWI2 no
        # observable), y el riesgo de intrusión queda capturado indirectamente por
        # J1 (storage) y J4 (costo desal/camiones). Ver paper_workflow_overview.md.
        super().__init__(n_vars, 5)
        self.types[:] = [Real(var_lo, var_hi) for _ in range(n_vars)]
        # Direcciones: todos minimize (J1 y J3 ya vienen negados en simulation())
        self.directions[:] = [Problem.MINIMIZE] * 5

    def evaluate(self, solution):
        P = np.array(solution.variables, dtype=float)
        J = self.pipe.simulation(P)
        solution.objectives[:] = list(J)
