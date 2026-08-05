# -*- coding: utf-8 -*-
"""
mlp_surrogate.py — Wrapper del WEAP-HydroMLP para uso desde el bridge DPS.

Provee dos modos de inferencia:

  predict_horizon(X)
      Forward end-to-end sobre toda la trayectoria (1872 weeks).
      Útil para Opción A (política estática) y para sanity checks.

  rollout_with_policy(initial_X, policy_fn, n_years, ...)
      Loop año a año. Después del warmup, cada año la policy_fn decide
      las acciones para los próximos 52 weeks basándose en el estado
      observado/predicho hasta ese punto. Implementa Opción B (adaptive).

Internamente cachea el checkpoint en CPU y reutiliza el modelo entre llamadas.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Categorías del manifest que corresponden al output de gw_net (524 vars)
GW_TYPES = {"GW", "GW_flux", "AP_wells", "Ag_wells"}

from weap_dps.config_weap import (
    CKPT_PATH, SCALERS_PATH, TRANSFORM_PARAMS_PATH, MANIFEST_PATH,
    WARMUP_WEEKS, WEEKS_PER_YEAR, ZARR_TEMPLATE_PATH,
)
from rdm_mlp.models.lightning_module import WEAPHydroMLPLightning

logger = logging.getLogger(__name__)


class MLPSurrogate:
    """Wrapper que mantiene el modelo cargado y los helpers de I/O."""

    def __init__(self, ckpt_path: Path | None = None,
                 device: str = "cpu",
                 manifest_path: Path | None = None,
                 scalers_path: Path | None = None,
                 transform_params_path: Path | None = None):
        ckpt_path = ckpt_path or CKPT_PATH
        manifest_path = manifest_path or MANIFEST_PATH
        scalers_path = scalers_path or SCALERS_PATH
        transform_params_path = transform_params_path or TRANSFORM_PARAMS_PATH

        logger.info("Loading checkpoint: %s", ckpt_path)
        self.model = WEAPHydroMLPLightning.load_from_checkpoint(
            str(ckpt_path), map_location=device,
        )
        self.model.eval()
        self.device = torch.device(device)

        # Hparams nested
        self.hp = self.model.hparams
        self.n_x = self.hp["model"]["n_x"]
        self.n_gw = self.hp["model"]["n_gw"]
        self.n_surface = self.hp["model"]["n_surface"]
        self.gw_lag_map = self.hp["model"].get("gw_lag_map", {})

        # Scalers (para denormalizar predicciones)
        scal = np.load(scalers_path)
        self.x_mean = scal["X_mean"] if "X_mean" in scal.files else scal.get("x_mean")
        self.x_std  = scal["X_std"]  if "X_std"  in scal.files else scal.get("x_std")
        self.y_mean = scal["Y_mean"] if "Y_mean" in scal.files else scal.get("y_mean")
        self.y_std  = scal["Y_std"]  if "Y_std"  in scal.files else scal.get("y_std")

        # Transforms (log/arcsinh por variable) — son por variable en orden RAW.
        # Hay que filtrar con y_keep_indices para mapear al orden de Y_filtered.
        tp = np.load(transform_params_path, allow_pickle=True)
        self.transform_alpha = float(tp.get("alpha", 0.1))

        raw_methods_y = tp.get("transform_method_y")
        y_keep = tp.get("y_keep_indices")
        if raw_methods_y is not None and y_keep is not None:
            # Orden filtrado (n_targets_filtered = n_gw + n_surface)
            self.transform_methods_y_filt = np.array(
                [str(raw_methods_y[i]) for i in y_keep],
                dtype=object,
            )
        else:
            self.transform_methods_y_filt = None

        # X transform methods filtrados por x_keep_indices (orden = X_filtered)
        raw_methods_x = tp.get("transform_method_x")
        x_keep = tp.get("x_keep_indices")
        if raw_methods_x is not None and x_keep is not None:
            self.transform_methods_x_filt = np.array(
                [str(raw_methods_x[i]) for i in x_keep],
                dtype=object,
            )
        else:
            self.transform_methods_x_filt = None

        # Target names — split en GW (524) y Surface (142) según Type.
        # Además, separa el array de transform_methods en GW vs Surface
        # con el mismo orden que el modelo entrega gw_pred/surf_pred.
        (self.target_names_gw, self.target_names_surf,
         self.transform_methods_gw, self.transform_methods_surf) = \
            self._load_target_names_and_methods(manifest_path)

        logger.info("Loaded MLP: n_x=%d  n_gw=%d  n_surface=%d  device=%s",
                    self.n_x, self.n_gw, self.n_surface, device)
        logger.info("  target_names: GW=%d  Surface=%d",
                    len(self.target_names_gw), len(self.target_names_surf))

    def _load_target_names_and_methods(self, manifest_path: Path):
        """
        Carga nombres + scalers + transform_methods para GW y Surface,
        respetando el orden EXACTO del output del modelo.

        Mismo patrón que `evaluate_recursive.py` del repo del modelo:
          - gw_idx_filt   = posiciones (en Y_filtered 0..665) de las vars GW
          - surface_idx_filt = el resto
          - scalers y transforms se indexan por estas posiciones, NO con
            slicing [:n_gw] (que era el bug original).

        Returns
        -------
        (gw_names, surf_names, gw_methods, surf_methods)
        Además rellena self.gw_idx_filt, self.surface_idx_filt,
        self.y_mean_gw, self.y_std_gw, self.y_mean_surf, self.y_std_surf.
        """
        # Fuente preferida: los índices que dejó `extract_data.py` en el
        # template, calculados con la MISMA build_indices que usa el DataModule.
        # Re-derivarlos del orden de filas del manifest NO sirve: ese espacio
        # (filas role=target) no coincide con las columnas de Y_filtered
        # (685 vs 677) y produce IndexError al indexar los scalers.
        gw_idx_filt = surf_idx_filt = None
        gw_names = surf_names = None
        try:
            tpl = np.load(ZARR_TEMPLATE_PATH, allow_pickle=True)
            if "gw_idx_filt" in tpl and "surface_idx_filt" in tpl:
                gw_idx_filt = np.asarray(tpl["gw_idx_filt"], dtype=int)
                surf_idx_filt = np.asarray(tpl["surface_idx_filt"], dtype=int)
                if "target_names_filtered" in tpl:
                    tn = list(tpl["target_names_filtered"])
                    gw_names = [tn[i] for i in gw_idx_filt]
                    surf_names = [tn[i] for i in surf_idx_filt]
                logger.info("Índices GW/Surface tomados del template "
                            "(n_gw=%d, n_surface=%d)", len(gw_idx_filt), len(surf_idx_filt))
        except Exception as exc:      # noqa: BLE001
            logger.warning("No se pudieron leer los índices del template (%s)", exc)

        df = pd.read_csv(manifest_path)
        targets = df[df["role"] == "target"].reset_index(drop=True)
        if gw_idx_filt is None:       # fallback legacy (layout antiguo)
            gw_mask = targets["Type"].isin(GW_TYPES).to_numpy()
            gw_idx_filt   = np.where(gw_mask)[0]
            surf_idx_filt = np.where(~gw_mask)[0]
            gw_names   = targets.loc[gw_mask,  "column"].tolist()
            surf_names = targets.loc[~gw_mask, "column"].tolist()
        if gw_names is None:
            gw_names   = [f"gw_{i}"   for i in gw_idx_filt]
            surf_names = [f"surf_{i}" for i in surf_idx_filt]

        # Scalers indexados por filt_idx
        if self.y_mean is not None and self.y_std is not None:
            self.y_mean_gw   = np.asarray(self.y_mean)[gw_idx_filt]
            self.y_std_gw    = np.asarray(self.y_std)[gw_idx_filt]
            self.y_mean_surf = np.asarray(self.y_mean)[surf_idx_filt]
            self.y_std_surf  = np.asarray(self.y_std)[surf_idx_filt]
        else:
            self.y_mean_gw = self.y_std_gw = None
            self.y_mean_surf = self.y_std_surf = None

        # Transform methods indexados por filt_idx
        if self.transform_methods_y_filt is not None:
            gw_methods   = np.asarray(self.transform_methods_y_filt)[gw_idx_filt]
            surf_methods = np.asarray(self.transform_methods_y_filt)[surf_idx_filt]
        else:
            gw_methods   = np.array(["none"] * len(gw_names),   dtype=object)
            surf_methods = np.array(["none"] * len(surf_names), dtype=object)

        self.gw_idx_filt      = gw_idx_filt
        self.surface_idx_filt = surf_idx_filt

        for nm, names, n_exp in (("GW",      gw_names,   self.n_gw),
                                  ("Surface", surf_names, self.n_surface)):
            if len(names) != n_exp:
                logger.warning("Mismatch %s: manifest=%d, modelo=%d",
                                nm, len(names), n_exp)

        return gw_names, surf_names, gw_methods, surf_methods

    # ─────────────────────────────────────────────────────────────────
    # Normalización de inputs (para inyectar acciones/q en X_tensor)
    # ─────────────────────────────────────────────────────────────────
    def normalize_x_value(self, raw_value: float, col_idx: int) -> float:
        """
        Convierte un valor RAW a su forma normalizada para inyectar en X_tensor.

        Aplica:
          1. transform (log/arcsinh/none) usando transform_methods_x_filt[col_idx]
          2. z-score: (val - x_mean[col_idx]) / x_std[col_idx]

        Args
        ----
        raw_value : valor en unidades físicas (0/1 para act_*; Mm³/yr para q_*)
        col_idx   : índice de la columna en X_filtered (orden del modelo)

        Returns
        -------
        valor normalizado, listo para inyectar en X_tensor[..., col_idx]
        """
        if self.x_mean is None or self.x_std is None:
            return float(raw_value)

        # 1. Transform
        if self.transform_methods_x_filt is not None and col_idx < len(self.transform_methods_x_filt):
            method = str(self.transform_methods_x_filt[col_idx])
        else:
            method = "none"
        alpha = self.transform_alpha
        if method == "log":
            # log(y + alpha). Para act_X=0 → log(alpha) ≈ -2.3 (válido)
            transformed = float(np.log(max(raw_value + alpha, 1e-12)))
        elif method == "arcsinh":
            transformed = float(np.arcsinh(raw_value / alpha))
        else:
            transformed = float(raw_value)

        # 2. z-score
        mean = float(self.x_mean[col_idx])
        std  = float(self.x_std[col_idx]) if self.x_std[col_idx] > 1e-12 else 1.0
        return (transformed - mean) / std

    # ─────────────────────────────────────────────────────────────────
    # Inferencia
    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_horizon(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward end-to-end sobre toda la trayectoria.

        Args
        ----
        X : np.ndarray  (T, n_x)  — ya NORMALIZADO (igual que durante training)

        Returns
        -------
        gw_pred_norm      : (T, n_gw)
        surface_pred_norm : (T, n_surface)
        """
        assert X.ndim == 2 and X.shape[1] == self.n_x, \
            f"X shape mismatch: got {X.shape}, expected (T, {self.n_x})"
        x_t = torch.tensor(X[None, ...], dtype=torch.float32, device=self.device)
        gw, surf = self.model.model.forward_sequence(
            x_t, warmup_steps=WARMUP_WEEKS,
        )
        return gw[0].cpu().numpy(), surf[0].cpu().numpy()

    @torch.no_grad()
    def rollout_with_policy(
        self,
        X_template: np.ndarray,
        policy_fn: Callable[[dict, int], dict],
        n_years: int,
        action_col_idx: dict,
        spin_up_years: int = 0,
    ) -> dict:
        """
        Rollout O(T) año a año con decisiones de política adaptive (Opción B).

        Implementación rápida: una sola pasada por todos los timesteps,
        emulando `forward_sequence` del modelo (lag-replacement GW desde
        el buffer de predicciones), pero ANTES de cada año-boundary aplica
        la decisión de política para los próximos 52 weeks.

        ~10-20 segundos por rollout (vs ~4 min de la versión que re-predecía
        toda la trayectoria por cada año).
        """
        T, F = X_template.shape
        device = self.device
        n_gw, n_surf = self.n_gw, self.n_surface

        # 1. Construir X y forzar acciones a 0 (NORMALIZADO) durante warmup + spin-up
        X = X_template.copy()
        pre_decision_end = WARMUP_WEEKS + spin_up_years * WEEKS_PER_YEAR
        # Precomputar el valor normalizado de 0 para cada columna de acción
        zero_norm = {nm: self.normalize_x_value(0.0, idx)
                     for nm, idx in action_col_idx.items()}
        for nm, idx in action_col_idx.items():
            X[:pre_decision_end, idx] = zero_norm[nm]

        X_tensor = torch.tensor(X[None, ...], dtype=torch.float32, device=device)

        # 2. Precomputar triples (X_col_idx, gw_var_idx, lag_offset) para el
        #    lag-replacement. Lo hago una vez fuera del loop temporal.
        lag_triples = self._get_lag_triples()

        # 3. Allocate output tensors
        gw_preds   = torch.zeros((1, T, n_gw),   device=device, dtype=torch.float32)
        surf_preds = torch.zeros((1, T, n_surf), device=device, dtype=torch.float32)

        # 4. Loop temporal único
        inner_model = self.model.model    # el WEAP_HydroMLP_Recursive
        actions_history = []
        policy_states = []
        year_idx = 0

        for t in range(T):
            # ── Decisión de política en cada año-boundary post spin-up ──
            if (t >= pre_decision_end
                and (t - pre_decision_end) % WEEKS_PER_YEAR == 0
                and year_idx < n_years):
                state = self._extract_state(
                    gw_preds[0].cpu().numpy(),
                    surf_preds[0].cpu().numpy(),
                    t_end_observed=t,
                )
                actions = policy_fn(state, year_idx)
                policy_states.append(state)
                actions_history.append(actions)
                year_idx += 1
                # Pisar las próximas 52 semanas (o lo que reste hasta T)
                # IMPORTANTE: normalizar el valor raw antes de inyectar en X_tensor
                t_end = min(t + WEEKS_PER_YEAR, T)
                for nm, idx in action_col_idx.items():
                    if nm in actions:
                        norm_val = self.normalize_x_value(float(actions[nm]), idx)
                        X_tensor[0, t:t_end, idx] = norm_val

            # ── Construir x_t (clon para no mutar X_tensor original) ──
            x_t = X_tensor[:, t, :].clone()

            # ── Lag-replacement vectorizado (solo después del warmup) ──
            if t >= WARMUP_WEEKS and self._lag_cols_t is not None:
                # t_lag por triple
                t_lag = t - self._lag_offs_t                  # (n_triples,)
                mask = (t_lag >= WARMUP_WEEKS) & (self._lag_vars_t < n_gw)
                if mask.any():
                    valid_cols   = self._lag_cols_t[mask]
                    valid_vars   = self._lag_vars_t[mask]
                    valid_t_lags = t_lag[mask]
                    # gw_preds shape (1, T, n_gw) — indexar (0, valid_t_lags, valid_vars)
                    replacements = gw_preds[0, valid_t_lags, valid_vars]
                    x_t[0, valid_cols] = replacements

            # ── Forward step ──
            gw_t, surf_t = inner_model(x_t)
            gw_preds[:,   t, :] = gw_t
            surf_preds[:, t, :] = surf_t

        # 5. Return como numpy
        return {
            "gw":      gw_preds[0].cpu().numpy(),
            "surface": surf_preds[0].cpu().numpy(),
            "X_used":  X_tensor[0].cpu().numpy(),
            "actions_history": np.array(
                [[a.get(k, 0.0) for k in action_col_idx] for a in actions_history],
                dtype=np.float32,
            ),
            "policy_states": policy_states,
        }

    def _get_lag_triples(self) -> list[tuple[int, int, int]]:
        """
        Convierte gw_lag_map ({'gw{var}_lag{offset}': col_idx}) en una lista
        de (col_idx, var_idx, lag_offset) iterable de una vez. Cacheado.
        """
        if hasattr(self, "_lag_triples_cache"):
            return self._lag_triples_cache
        triples = []
        for key, col_idx in self.gw_lag_map.items():
            if "_lag" not in key:
                continue
            try:
                var_part, lag_part = key.split("_lag")
                var_idx = int(var_part.replace("gw", ""))
                lag_off = int(lag_part)
                triples.append((int(col_idx), var_idx, lag_off))
            except ValueError:
                continue
        self._lag_triples_cache = triples
        # Tensores para indexado vectorizado
        if triples:
            cols = torch.tensor([t[0] for t in triples], dtype=torch.long, device=self.device)
            vars_ = torch.tensor([t[1] for t in triples], dtype=torch.long, device=self.device)
            lags = torch.tensor([t[2] for t in triples], dtype=torch.long, device=self.device)
            self._lag_cols_t = cols
            self._lag_vars_t = vars_
            self._lag_offs_t = lags
        else:
            self._lag_cols_t = self._lag_vars_t = self._lag_offs_t = None
        logger.info("Lag triples cacheadas: %d entradas", len(triples))
        return triples

    # ─────────────────────────────────────────────────────────────────
    # State extraction
    # ─────────────────────────────────────────────────────────────────
    def _extract_state(self, gw_norm: np.ndarray, surf_norm: np.ndarray,
                       t_end_observed: int, lookback: int = 52) -> dict:
        """
        Resumen del estado del sistema en t=t_end_observed (decisión anual).
        Por defecto usa los últimos 52 weeks (1 año) para promediar.

        Returns dict con escalares: gw_storage_avg, gw_storage_min,
        surface_avg, recent_trend.
        """
        t0 = max(0, t_end_observed - lookback)
        gw_recent   = gw_norm[t0:t_end_observed]
        surf_recent = surf_norm[t0:t_end_observed]
        return {
            "gw_storage_avg": float(np.nanmean(gw_recent)),
            "gw_storage_min": float(np.nanmin(gw_recent)),
            "surface_avg":    float(np.nanmean(surf_recent)),
            "gw_trend":       float(np.nanmean(gw_recent[-13:]) - np.nanmean(gw_recent[:13]))
                              if gw_recent.shape[0] >= 26 else 0.0,
            "year_idx":       t_end_observed // WEEKS_PER_YEAR,
        }

    # ─────────────────────────────────────────────────────────────────
    # Denormalización
    # ─────────────────────────────────────────────────────────────────
    def denormalize_y(self, y_norm: np.ndarray, kind: str = "gw") -> np.ndarray:
        """
        Aplica la inversa COMPLETA (z-score + transform) por columna.

        El modelo predice gw_pred (T, n_gw) y surf_pred (T, n_surface) en
        espacio normalizado. La normalización original fue:
            1. transform: log(y + alpha) o arcsinh(y / alpha) o identidad
            2. z-score: (y_t - mean) / std

        Para invertir:
            1. y_unstd = y_norm * std + mean
            2. y_raw = inverse_transform(y_unstd, method)
                       - log:     exp(y_unstd) - alpha
                       - arcsinh: sinh(y_unstd) * alpha
                       - none:    y_unstd

        kind ∈ {'gw', 'surface'}
        """
        if self.y_mean is None:
            return y_norm

        # IMPORTANTE: los scalers (y_mean/y_std) y los transform_methods estan en
        # orden de Y_filtered (manifest, mezcla GW+Surface). NO se pueden cortar
        # con [:n_gw] / [n_gw:]. Hay que indexar con gw_idx_filt / surface_idx_filt
        # (calculados en _load_target_names_and_methods).
        if kind == "gw":
            mean    = self.y_mean_gw
            std     = self.y_std_gw
            methods = self.transform_methods_gw
        elif kind == "surface":
            mean    = self.y_mean_surf
            std     = self.y_std_surf
            methods = self.transform_methods_surf
        else:
            raise ValueError(f"kind debe ser 'gw' | 'surface', recibí '{kind}'")

        if y_norm.shape[-1] != mean.shape[-1]:
            raise ValueError(
                f"y_norm tiene {y_norm.shape[-1]} columnas pero los scalers '{kind}' "
                f"tienen {mean.shape[-1]}. Revisa el slicing de Y_filtered."
            )

        # 1. z-score inverso
        y_unstd = y_norm * std + mean

        # 2. inversa de log / arcsinh por columna
        if methods is None or len(methods) != mean.shape[-1]:
            return y_unstd

        alpha = self.transform_alpha
        out = np.empty_like(y_unstd)
        for j, m in enumerate(methods):
            col = y_unstd[..., j]
            m = str(m)
            if m == "log":
                # clipear para evitar overflow en exp
                col_clip = np.clip(col, -30, 30)
                # No-negatividad: log se aplicó solo a variables positivas
                # (transmisión, suministro) → el valor físico no puede ser < 0.
                out[..., j] = np.maximum(np.exp(col_clip) - alpha, 0.0)
            elif m == "arcsinh":
                out[..., j] = np.sinh(col) * alpha
            else:
                out[..., j] = col
        return out
