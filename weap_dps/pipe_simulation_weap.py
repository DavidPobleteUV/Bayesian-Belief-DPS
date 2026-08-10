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
    POLICY_STATE_FEATURES, N_STATE_FEATURES,
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

        # Los escaladores vienen en el espacio de X_filtered; el template ya
        # está recortado al sub-conjunto del manifest. Si no se alinean, cada
        # acción se normaliza con el escalador de otra columna.
        n_cols = self.X_template.shape[1]
        if self.surrogate.x_mean is not None and len(self.surrogate.x_mean) != n_cols:
            x_idx = data.get("x_idx_filt")
            if x_idx is None:
                raise RuntimeError(
                    f"El template tiene {n_cols} columnas y los escaladores "
                    f"{len(self.surrogate.x_mean)}, pero el template no trae "
                    f"'x_idx_filt' para alinearlos. Regenera el template con "
                    f"`python weap_dps/extract_data.py` (necesita el zarr completo).")
            self.surrogate.subset_x_scalers(x_idx)

        # Lookup de columnas de acción
        self.action_col_idx = build_action_col_idx(self.feature_names)

        # Estado ampliado de la política: antes solo veía agregados de GW y el
        # calendario, así que no podía reaccionar a J2/J3/J4/J6 — los objetivos
        # que de hecho varían a lo largo del frente.
        self.surrogate.configure_policy_state(self.feature_names)
        self.ap_town_order = list(getattr(self.surrogate, "_st_towns", []))
        self._ap_dem_cols = list(getattr(self.surrogate, "_st_dem_x", []))
        self.surrogate.set_truck_columns(self._truck_link_cols())

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

    def _ap_demand(self, X_scen: np.ndarray) -> np.ndarray | None:
        """Demanda AP por pueblo (m³/s), desnormalizada desde el escenario.

        J51/J52 son razones déficit/demanda, y la demanda es INPUT del modelo,
        no target: hay que sacarla de X. Cada escenario tiene su propia demanda
        (los corners de población la escalan), así que se recalcula por escenario.
        """
        if not self._ap_dem_cols:
            return None
        return self.surrogate.denormalize_x_cols(X_scen, self._ap_dem_cols)

    def _truck_link_cols(self) -> list[int]:
        """Índices de los links de camiones en target_names_surf.

        Los camiones son la fuente cara de emergencia (8000 CLP/m³): su peso en
        el suministro es el mejor indicador de estrés que la política puede leer
        para anticipar J4.
        """
        import pandas as pd
        from weap_dps.config_weap import TOWN_SOURCE_COST_CSV
        if not Path(TOWN_SOURCE_COST_CSV).exists():
            logger.warning("Sin %s: truck_frac quedará en 0", TOWN_SOURCE_COST_CSV)
            return []
        cm = pd.read_csv(TOWN_SOURCE_COST_CSV)
        cm["withdrawal_node"] = cm["withdrawal_node"].astype(str).str.strip()
        nodes = set(cm[cm["source_type"] == "Camiones"]["withdrawal_node"])
        cols = []
        # del surrogate, no de self: self.target_names_surf se asigna más abajo
        for i, n in enumerate(self.surrogate.target_names_surf):
            if not (n.startswith("AP_TransmissionLinks__") and "_to_" in n):
                continue
            s = n.split("__", 1)[1].rsplit("_to_", 1)[0].replace("Transmission_Link_from_", "")
            if s.startswith("Withdrawal_Node_") and s.replace("Withdrawal_Node_", "") in nodes:
                cols.append(i)
        return cols

    # ─── Policy NN del DPS ──────────────────────────────────────────────
    def _build_policy_from_params(self, P: np.ndarray,
                                  n_state_features: int = N_STATE_FEATURES):
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

            # Los flags built_* salen de este closure, no de _extract_state:
            # son estado de la POLITICA (que ya construyo), no del sistema.
            # built_X  ->  act_X, que es como se llaman en ACTION_NAMES_INFRA
            s = np.array([built.get("act_" + f[6:], 0.0) if f.startswith("built_")
                          else state_dict.get(f, 0.0)
                          for f in POLICY_STATE_FEATURES], dtype=float)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
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
                ap_demand_m3s=self._ap_demand(X_scen),
                ap_town_order=self.ap_town_order,
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
             J_mean[4],                           # J51 semanas de falla, promedio por pueblo
             J_mean[5],                           # J52 peor año (déficit/demanda)
             J_mean[6],                           # J6 salinidad costera (min)
        ]
        return tuple(J)
