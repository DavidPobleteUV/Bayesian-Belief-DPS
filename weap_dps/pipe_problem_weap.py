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
        # 6 objetivos (J1..J6). J6 (salinidad costera) fue REINCORPORADO: el
        # modelo iter0_900 predice Z_value (cota de la interfaz SWI2) en los 12
        # pozos costeros AP de Q09, así que la intrusión salina ya es observable
        # y la salinidad se deriva de forma determinista (ver
        # cost_calculator.j6_coastal_salinity / salinity_from_zvalue).
        # Antes se había eliminado porque sin zeta solo había discriminación
        # gruesa y el riesgo quedaba capturado de forma indirecta por J1 y J4.
        super().__init__(n_vars, 6)
        self.types[:] = [Real(var_lo, var_hi) for _ in range(n_vars)]
        # Direcciones: todos minimize (J1 y J3 ya vienen negados en simulation())
        self.directions[:] = [Problem.MINIMIZE] * 6

    def evaluate(self, solution):
        P = np.array(solution.variables, dtype=float)
        J = self.pipe.simulation(P)
        solution.objectives[:] = list(J)
