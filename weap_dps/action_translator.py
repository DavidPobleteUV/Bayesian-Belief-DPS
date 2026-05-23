# -*- coding: utf-8 -*-
"""
action_translator.py — Convierte salidas de la policy NN del DPS a las columnas
del input MLP. Las 5 acciones son BINARIAS PURAS: el policy emite 5 valores en
[0,1] que se umbralizan a 0/1; al activarse una acción se inyecta su cantidad
canónica (CANONICAL_Q), porque el MLP solo vio un valor de q por acción.

Devuelve un dict {col_name: value} con las 5 binarias + sus 5 q canónicas,
listo para inyectar en X.
"""

from __future__ import annotations

import numpy as np

from weap_dps.config_weap import (
    ACTION_NAMES_BINARY, ACTION_NAMES_QUANTITY, CANONICAL_Q,
)

# q canónico indexado por la binaria correspondiente (mismo orden).
_Q_BY_BINARY = dict(zip(ACTION_NAMES_BINARY, ACTION_NAMES_QUANTITY))


def policy_output_to_actions(pi_out: np.ndarray,
                             enforce_restrictions: bool = True) -> dict[str, float]:
    """
    Args
    ----
    pi_out : np.ndarray (5,) — output crudo de la policy NN, cada componente en
                [0,1] = probabilidad de activar la acción binaria i.

    enforce_restrictions : si True, aplica restricciones del catálogo:
        R1: completa subsume costera → si completa=1 entonces costera=0.

    Returns
    -------
    actions : dict {col_name: float} con act_* (0/1) y q_* (canónico o 0).
    """
    pi_out = np.asarray(pi_out, dtype=float).flatten()
    n = len(ACTION_NAMES_BINARY)
    assert pi_out.size >= n, f"Esperaba {n}+ outputs, recibí {pi_out.size}"

    actions: dict[str, float] = {}

    # Binarias: umbralizar; q = canónico si on, 0 si off
    for i, bin_name in enumerate(ACTION_NAMES_BINARY):
        on = float(pi_out[i] > 0.5)
        actions[bin_name] = on
        q_name = _Q_BY_BINARY[bin_name]
        actions[q_name] = CANONICAL_Q[q_name] if on else 0.0

    if enforce_restrictions:
        # R1: completa subsume costera
        if actions["act_desalacion_completa"] == 1 and actions["act_desalacion_costera"] == 1:
            actions["act_desalacion_costera"] = 0.0
            actions["q_desalacion_costera"]   = 0.0

    return actions


def actions_to_policy_output(actions: dict[str, float]) -> np.ndarray:
    """Inverso (para testing/sanity): 5 binarias."""
    out = np.zeros(len(ACTION_NAMES_BINARY), dtype=float)
    for i, name in enumerate(ACTION_NAMES_BINARY):
        out[i] = float(actions.get(name, 0.0))
    return out


# ─── Indexado por nombre en el feature vector del MLP ───────────────────
def build_action_col_idx(feat_names: list[str]) -> dict[str, int]:
    """Devuelve dict {action_name: column_idx_en_X} para los 6 inputs de política."""
    out = {}
    for name in ACTION_NAMES_BINARY + ACTION_NAMES_QUANTITY:
        if name in feat_names:
            out[name] = feat_names.index(name)
        else:
            raise KeyError(f"Acción '{name}' no encontrada en feature_names del MLP")
    return out
