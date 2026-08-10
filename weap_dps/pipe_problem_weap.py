# -*- coding: utf-8 -*-
"""
pipe_problem_weap.py — Wrapper Platypus para el problema multiobjetivo
de Quilimari (7 objetivos: J1, J2, J3, J4, J51, J52, J6).
"""

from __future__ import annotations

import logging

import numpy as np
from platypus import Problem, Real

from weap_dps.config_weap import N_STATE_FEATURES
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
            M=pipe.policy_M, K=pipe.policy_K, N=N_STATE_FEATURES,
        )
        # 6 objetivos (J1..J6). J6 (salinidad costera) fue REINCORPORADO: el
        # modelo iter0_900 predice Z_value (cota de la interfaz SWI2) en los 12
        # pozos costeros AP de Q09, así que la intrusión salina ya es observable
        # y la salinidad se deriva de forma determinista (ver
        # cost_calculator.j6_coastal_salinity / salinity_from_zvalue).
        # Antes se había eliminado porque sin zeta solo había discriminación
        # gruesa y el riesgo quedaba capturado de forma indirecta por J1 y J4.
        # 7 objetivos: J5 se partió en J51 (semanas de falla, promedio por
        # pueblo) y J52 (peor año). El J5 original sumaba el déficit de todos
        # los pueblos contra un umbral absoluto de 100 m3/semana y se saturaba.
        super().__init__(n_vars, 7)
        self.types[:] = [Real(var_lo, var_hi) for _ in range(n_vars)]
        # Direcciones: todos minimize (J1 y J3 ya vienen negados en simulation())
        self.directions[:] = [Problem.MINIMIZE] * 7

    def evaluate(self, solution):
        P = np.array(solution.variables, dtype=float)
        J = self.pipe.simulation(P)
        solution.objectives[:] = list(J)
