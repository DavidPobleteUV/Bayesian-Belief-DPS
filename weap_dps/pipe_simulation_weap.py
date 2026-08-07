# -*- coding: utf-8 -*-
"""
pipe_simulation_weap.py — Equivalente del `Pipe.simulation()` original del
paper Bayesian DPS, pero usando WEAP-HydroMLP como modelo de sistema.

Implementa Opción B: la policy NN decide acciones cada año durante el
horizonte de decisión. Devuelve los 5 objetivos J1..J5.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import (
    SPIN_UP_YEARS, DECISION_YEARS, WARMUP_WEEKS, WEEKS_PER_YEAR,
    N_WEEKS_HORIZON, j4_calibration_factor, DPS_WATERFALL,
)
from weap_dps.mlp_surrogate import MLPSurrogate
from weap_dps.action_translator import (
    policy_output_to_actions, build_action_col_idx,
    init_built_state, update_built_state,
    ACTION_NAMES_BINARY, ACTION_NAMES_QUANTITY, CANONICAL_Q,
)
from weap_dps.action_translator import _Q_BY_BINARY
from weap_dps.cost_calculator import compute_objectives

logger = logging.getLogger(__name__)


class PipeWEAP:
    """
    Wrapper de simulación que mantiene el surrogate cargado y ofrece
    `simulation(policy_params)` retornando los 5 objetivos.

    Parameters
    ----------
    template_path : Path  — `.npz` con X_template (1872, 611) y feature_names
    scenarios     : list[np.ndarray]  — lista de X completos por escenario (ya
                    con clima + áreas + población aplicados), o None para usar
                    solo el template (1 escenario).
    """

    def __init__(self,
                 template_path: Path,
                 scenarios: list[np.ndarray] | None = None,
                 policy_arch: tuple[int, int] | None = None):
        # Cargar surrogate
        self.surrogate = MLPSurrogate()
        # K se deriva del catálogo de acciones vigente: si se fija a mano y no
        # coincide, las salidas sobrantes de la política quedan MUERTAS
        # (action_translator solo lee las primeras len(ACTION_NAMES_BINARY)) y
        # NSGA-II gasta presupuesto optimizando variables que no hacen nada.
        if policy_arch is None:
            policy_arch = (14, len(ACTION_NAMES_BINARY))
        self.policy_M, self.policy_K = policy_arch   # hidden, output dim
        if self.policy_K != len(ACTION_NAMES_BINARY):
            logger.warning("policy_K=%d != %d acciones del catálogo: %d salidas "
                           "de la política quedarán sin uso",
                           self.policy_K, len(ACTION_NAMES_BINARY),
                           self.policy_K - len(ACTION_NAMES_BINARY))

        # Cargar template
        data = np.load(template_path, allow_pickle=True)
        self.X_template = data["X"].astype(np.float32)
        self.feature_names = list(data["feature_names"])
        self.scenarios = scenarios if scenarios is not None else [self.X_template]

        # Lookup de columnas de acción
        self.action_col_idx = build_action_col_idx(self.feature_names)

        # Target names — vienen del surrogate (ya cargados desde el manifest)
        self.target_names_gw   = self.surrogate.target_names_gw
        self.target_names_surf = self.surrogate.target_names_surf

        # .3 waterfall: allocator construido una sola vez (mapping town->links)
        self.waterfall = None
        if DPS_WATERFALL:
            from weap_dps.waterfall_alloc import WaterfallAllocator
            self.waterfall = WaterfallAllocator(
                self.surrogate, self.feature_names, self.target_names_surf,
            )
            logger.info(".3 WATERFALL enabled — J4 uses well-anchored cascade "
                        "(%d towns mapped)", len(self.waterfall.towns))

    # ─── Policy NN del DPS ──────────────────────────────────────────────
    def _build_policy_from_params(self, P: np.ndarray, n_state_features: int = 4):
        """
        Construye una mini-NN del policy con los parámetros P (flat array).
        Arquitectura: state (n_state_features) → hidden (M) → sigmoid(K).

        Aquí K = 5 (5 acciones binarias puras; q se inyecta canónico).
        """
        M, K = self.policy_M, self.policy_K
        N = n_state_features
        expected_size = N * M + M + M * K + K
        assert len(P) == expected_size, \
            f"P tiene {len(P)} params, esperaba {expected_size}"
        W1 = P[:N*M].reshape(N, M)
        b1 = P[N*M:N*M+M]
        W2 = P[N*M+M:N*M+M+M*K].reshape(M, K)
        b2 = P[N*M+M+M*K:]

        # Estado de obras construidas. La política re-decide cada año SIN
        # memoria, así que sin esto puede encender la desaladora en 2027 y
        # apagarla en 2028: el CAPEX se cobra igual (se detecta la primera
        # activación) pero la columna q vuelve a 0 y el surrogate deja de
        # entregar agua. Resultado: se paga la planta y no se usa.
        built = init_built_state()

        def policy_fn(state_dict: dict, year_idx: int) -> dict:
            # El mismo policy_fn se reutiliza en los N escenarios del ensamble;
            # cada rollout arranca en year_idx=0 y debe partir sin obras.
            nonlocal built
            if year_idx == 0:
                built = init_built_state()

            s = np.array([state_dict["gw_storage_avg"],
                          state_dict["gw_storage_min"],
                          state_dict["gw_trend"],
                          year_idx / 35.0], dtype=float)
            h = np.tanh(s @ W1 + b1)
            raw = h @ W2 + b2
            pi = 1.0 / (1.0 + np.exp(-raw))   # sigmoid → [0,1]
            actions = policy_output_to_actions(pi)

            # Irreversibilidad: una obra construida sigue operando aunque la
            # política la "apague". Solo aplica a ACTION_NAMES_INFRA; el
            # acuerdo es administrativo y sí puede revertirse año a año.
            built = update_built_state(built, actions)
            for name, is_built in built.items():
                if is_built and not actions.get(name, 0.0):
                    actions[name] = 1.0
                    q = _Q_BY_BINARY[name]
                    actions[q] = CANONICAL_Q[q]

            # Re-aplicar R1: el forzado de arriba puede revivir la costera un
            # año después de que se construyó la completa, que la subsume.
            # (built["costera"] se mantiene en 1: la obra existe y su CAPEX ya
            # se pagó en su año de activación; solo deja de operar.)
            if actions["act_desalacion_completa"] and actions["act_desalacion_costera"]:
                actions["act_desalacion_costera"] = 0.0
                actions["q_desalacion_costera"] = 0.0
            return actions
        return policy_fn

    @staticmethod
    def policy_param_size(M: int = 14, K: int = 5, N: int = 4) -> int:
        """Cantidad total de parámetros del policy NN."""
        return N * M + M + M * K + K

    # ─── Simulación ─────────────────────────────────────────────────────
    def simulation(self, P: np.ndarray) -> tuple[float, ...]:
        """
        Args
        ----
        P : array de parámetros del policy NN.

        Returns
        -------
        tupla (J1, J2, J3, J4, J5) con J1, J3 negados (para que NSGA
        siempre minimice).
        """
        policy_fn = self._build_policy_from_params(P)

        all_J = []
        for scen_idx, X_scen in enumerate(self.scenarios):
            result = self.surrogate.rollout_with_policy(
                X_template=X_scen,
                policy_fn=policy_fn,
                n_years=DECISION_YEARS,
                action_col_idx=self.action_col_idx,
                spin_up_years=SPIN_UP_YEARS,
            )
            # Desnormalizar
            gw_denorm   = self.surrogate.denormalize_y(result["gw"],     kind="gw")
            surf_denorm = self.surrogate.denormalize_y(result["surface"], kind="surface")

            # .3 waterfall: reemplaza desal/aduccion/pozo/camiones por la cascada
            # well-anclada (y anula Acuerdo) ANTES de costear J4.
            if self.waterfall is not None:
                surf_denorm = self.waterfall.apply(surf_denorm, result["X_used"])

            objs = compute_objectives(
                gw_denorm=gw_denorm,
                surf_denorm=surf_denorm,
                target_names_gw=self.target_names_gw,
                target_names_surf=self.target_names_surf,
                actions_history=result["actions_history"],
                action_names_order=ACTION_NAMES_BINARY + ACTION_NAMES_QUANTITY,
                decision_start_week=WARMUP_WEEKS + SPIN_UP_YEARS * WEEKS_PER_YEAR,
            )
            all_J.append(list(objs.values()))

        # Promedio entre escenarios
        J_mean = np.nanmean(np.array(all_J), axis=0)
        # NSGA minimiza → convertir J1 (maximizar storage) y J3 (maximizar valor) a min.
        # J6 (índice 5) REINCORPORADO: se deriva del Z_value predicho (ver
        # cost_calculator.j6_coastal_salinity). Ya se minimiza tal cual.
        # J4 se calibra: el surrogate SUB-PREDICE el volumen de respaldo, y el
        # sesgo crece con el número de acciones activas (1.17 con 0 acciones →
        # 2.07 con 3+). Un escalar único favorecería sistemáticamente a las
        # políticas con muchas acciones (las caras), así que el factor se elige
        # según cuántas acciones enciende ESTA política.
        n_act = 0
        ah = result.get("actions_history")
        if ah is not None and len(ah):
            ah = np.asarray(ah)
            n_bin = len(ACTION_NAMES_BINARY)
            # una acción cuenta como activa si se enciende en algún año
            n_act = int((ah[:, :n_bin] > 0.5).any(axis=0).sum())
        cal = j4_calibration_factor(n_act)

        J = [
            -J_mean[0],                           # J1 GW storage (max → neg)
             J_mean[1],                           # J2 unmet AP
            -J_mean[2],                           # J3 agri value (max → neg)
             J_mean[3] * cal,                     # J4 cost (calibrado por nº acciones)
             J_mean[4],                           # J5 weeks failure
             J_mean[5],                           # J6 salinidad costera (min)
        ]
        return tuple(J)
