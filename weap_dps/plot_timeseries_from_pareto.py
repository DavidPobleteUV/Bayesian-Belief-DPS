# -*- coding: utf-8 -*-
"""
plot_timeseries_from_pareto.py — Plotea series temporales de las variables
de interés para soluciones específicas del frente de Pareto.

Para cada solución seleccionada:
  1. Re-corre la policy NN sobre el horizonte completo (rollout_with_policy).
  2. Extrae las series temporales de variables clave (GW storage, Unmet AP,
     producción agrícola, costos por fuente).
  3. Plotea cada variable como una figura, con cada solución coloreada
     distinto y la leyenda mostrando los valores de los objetivos J1..J5.

Selección de soluciones:
  - extremes : 5 puntos extremos (1 por cada Jk minimizado)
  - spread   : N puntos uniformemente espaciados en el frente
  - all      : todos los puntos del frente (puede ser lento si son muchos)
  - first_N  : primeros N puntos del .dat

Uso:
    # 5 extremos (default)
    python weap_dps/plot_timeseries_from_pareto.py \
        --pareto runs_weap/pareto_iter01_combined_short.dat \
        --output_dir runs_weap/timeseries_iter01

    # 10 soluciones spread
    python weap_dps/plot_timeseries_from_pareto.py \
        --pareto runs_weap/pareto_iter01_combined_short.dat \
        --selection spread --n_runs 10 \
        --output_dir runs_weap/timeseries_iter01_spread10
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weap_dps.config_weap import (
    ZARR_TEMPLATE_PATH, WARMUP_WEEKS, SPIN_UP_YEARS, WEEKS_PER_YEAR,
    BASE_YEAR, ANALYSIS_HORIZON_Y, USD_CLP_RATE,
)
from weap_dps.pipe_simulation_weap import PipeWEAP

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [TS_PLOT] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Paleta de colores (Brewer Set1 + extras)
PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#FFD92F", "#A65628", "#F781BF",
    "#999999", "#000000", "#1B9E77", "#D95F02",
    "#7570B3", "#E7298A", "#66A61E", "#E6AB02",
]


# ─── Selección de soluciones ─────────────────────────────────────────────

def load_pareto(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["result"]   # [(variables, objectives), ...]


def select_solutions(sols, strategy: str, n_runs: int) -> list[int]:
    """
    Retorna los índices de las soluciones seleccionadas.

    'objectives' del .dat: [J1_neg, J2, J3_neg, J4, J5] (NSGA minimiza todos).
    """
    objs = np.array([o for _, o in sols], dtype=float)
    n_total = len(sols)

    if strategy == "extremes":
        # 5 mínimos (uno por objetivo en el espacio negado)
        idxs = []
        for k in range(objs.shape[1]):
            idx = int(np.argmin(objs[:, k]))
            if idx not in idxs:
                idxs.append(idx)
        return idxs[:n_runs] if n_runs > 0 else idxs

    if strategy == "all":
        return list(range(n_total))

    if strategy == "first_N":
        return list(range(min(n_runs, n_total)))

    if strategy == "spread":
        # Ordenar por J4 (costo NPV) y tomar n_runs uniformemente espaciados
        order = np.argsort(objs[:, 3])
        if n_runs >= n_total:
            return list(order)
        idx_in_order = np.linspace(0, n_total - 1, n_runs).astype(int)
        return list(order[idx_in_order])

    raise ValueError(f"strategy desconocida: {strategy}")


# ─── Re-evaluación de soluciones ─────────────────────────────────────────

def evaluate_solution(pipe: PipeWEAP, solution_vars: np.ndarray) -> dict:
    """Corre la policy y devuelve outputs denormalizados + actions_history."""
    policy_fn = pipe._build_policy_from_params(np.asarray(solution_vars, dtype=float))
    result = pipe.surrogate.rollout_with_policy(
        X_template=pipe.X_template,
        policy_fn=policy_fn,
        n_years=ANALYSIS_HORIZON_Y,
        action_col_idx=pipe.action_col_idx,
        spin_up_years=SPIN_UP_YEARS,
    )
    gw_denorm   = pipe.surrogate.denormalize_y(result["gw"],     kind="gw")
    surf_denorm = pipe.surrogate.denormalize_y(result["surface"], kind="surface")
    return {
        "gw_denorm":       gw_denorm,
        "surf_denorm":     surf_denorm,
        "actions_history": result["actions_history"],
    }


# ─── Extractores de variables ───────────────────────────────────────────

def _build_dates(n_steps: int):
    return pd.date_range("2015-01-05", periods=n_steps, freq="W-MON")


def _safe_indices(names: list[str], patterns: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if any(p in n for p in patterns)]


def extract_gw_storage_total(gw_denorm, target_names_gw) -> np.ndarray:
    """Suma de los 9 SHAC_storage_Acuifero a lo largo del tiempo."""
    cols = _safe_indices(target_names_gw,
        [f"SHAC_storage_Acuifero_Q0{i}_MF_m3" for i in range(1, 10)])
    if not cols:
        return None
    return gw_denorm[:, cols].sum(axis=1)


def extract_unmet_ap_total(surf_denorm, target_names_surf) -> np.ndarray:
    """
    Suma de AP_UnmetDemand_* en l/s (raw del MLP, m³/s × 1000).

    Clipea a 0 los valores ligeramente negativos que produce la inversa
    del log transform del MLP — sin significado físico.
    """
    cols = _safe_indices(target_names_surf,
                          ["AP_UnmetDemand", "Demanda No Atendida AP"])
    if not cols:
        return None
    raw_m3_s = np.maximum(surf_denorm[:, cols], 0.0)   # clip per-column
    return raw_m3_s.sum(axis=1) * 1000.0   # m³/s → l/s


def extract_agri_production_annual(surf_denorm, target_names_surf) -> np.ndarray:
    """
    Producción agrícola anual (short ton) — un valor por año hidrológico,
    expandido a step function semanal para mantener el shape (T,).

    Lógica:
      1. Para cada año (52 weeks consecutivos), tomar la MEDIA semanal
         por cultivo (los valores anuales están replicados 52x en el MLP).
      2. Sumar las 12 series (cultivos × zonas) → producción anual total.
      3. Re-expandir a (T,) replicando cada valor anual 52 veces.

    Devuelve una serie semanal "stepped" (constante por año), apta para
    overlay con otras series semanales.
    """
    # Solo Palto (explícito por consistencia con j3_agricultural_value)
    cols = [i for i, n in enumerate(target_names_surf)
            if "AGR_AnnualCropProduction" in n and ("Palto" in n or "palto" in n)]
    if not cols:
        return None
    T = surf_denorm.shape[0]
    n_years = T // WEEKS_PER_YEAR
    # Clip negativos (sin significado físico, vienen de la inversa del log)
    weekly = np.maximum(surf_denorm[:, cols], 0.0)      # (T, n_crops)

    # Annual production per year (suma sobre cultivos del MEAN sobre weeks)
    annual_total = np.zeros(n_years)
    for y in range(n_years):
        week_slice = weekly[y * WEEKS_PER_YEAR : (y + 1) * WEEKS_PER_YEAR]
        annual_means = np.nanmean(week_slice, axis=0)   # (n_crops,)
        annual_total[y] = np.nansum(annual_means)

    # Expandir a step function semanal
    stepped = np.repeat(annual_total, WEEKS_PER_YEAR)
    # Pad si quedan weeks sueltos al final
    if len(stepped) < T:
        pad = np.full(T - len(stepped), stepped[-1] if len(stepped) else 0.0)
        stepped = np.concatenate([stepped, pad])
    return stepped[:T]


# ─── Plotter genérico de series temporales ──────────────────────────────

def plot_series_overlay(
    series_per_sol: dict[int, np.ndarray],
    sol_metadata: dict[int, dict],
    title: str,
    ylabel: str,
    output_path: Path,
    annotate_with: tuple[str, ...] = ("J4_MUSD", "J2_lps"),
    decision_start_week: int = None,
    warmup_end_week: int = None,
):
    """
    Plot N series temporales sobreimpresas con el estilo del visualize_results
    del repo del modelo:
      - figsize wider (14, 5)
      - título en formato 'TOTAL_<var>_AllSHACs'
      - línea vertical roja punteada en warmup_end (donde arranca recursión)
      - línea vertical gris en decision_start (donde policy empieza a actuar)
      - leyenda con sol#X y métricas J*
      - notación científica en eje Y
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(14, 5))
    handles_scenarios = []
    for idx, (sol_idx, series) in enumerate(series_per_sol.items()):
        color = PALETTE[idx % len(PALETTE)]
        dates = _build_dates(len(series))
        meta = sol_metadata[sol_idx]
        annot = []
        for key in annotate_with:
            if key in meta:
                annot.append(f"{key}={meta[key]:.2g}")
        label = f"sol#{sol_idx}" + (f"  ({', '.join(annot)})" if annot else "")
        ax.plot(dates, series, color=color, linewidth=1.3, alpha=0.85)
        handles_scenarios.append(
            Line2D([0], [0], color=color, linewidth=2.0, label=label)
        )

    # Línea vertical en warmup end (donde MLP empieza a usar predicciones)
    if warmup_end_week is not None and 0 < warmup_end_week < len(dates):
        ax.axvline(dates[warmup_end_week], color="red", linestyle=":",
                   alpha=0.7, linewidth=1.2)
    # Línea vertical en decision_start (donde policy empieza)
    if decision_start_week is not None and 0 < decision_start_week < len(dates):
        ax.axvline(dates[decision_start_week], color="gray", linestyle="--",
                   alpha=0.6, linewidth=1.0)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    # Leyenda en dos cajas: Reference lines (vertical) + Scenarios
    ref_handles = []
    if warmup_end_week is not None:
        ref_handles.append(Line2D([0], [0], color="red", linestyle=":",
                                  linewidth=1.2, label="warmup end (recursión)"))
    if decision_start_week is not None:
        ref_handles.append(Line2D([0], [0], color="gray", linestyle="--",
                                  linewidth=1.0, label="inicio decisiones DPS"))
    if ref_handles:
        leg_ref = ax.legend(handles=ref_handles, loc="upper left",
                             fontsize=8, framealpha=0.9, title="Reference",
                             title_fontsize=8)
        ax.add_artist(leg_ref)
    ax.legend(handles=handles_scenarios, loc="upper right",
              fontsize=8, framealpha=0.9, title="Scenarios",
              title_fontsize=8, ncol=1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pareto",      type=Path, required=True)
    p.add_argument("--selection",   choices=["extremes", "spread", "all", "first_N"],
                   default="extremes")
    p.add_argument("--n_runs",      type=int, default=5,
                   help="Cantidad a graficar (ignorado si selection=all/extremes)")
    p.add_argument("--output_dir",  type=Path, default=Path("runs_weap/timeseries"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar Pareto + seleccionar
    sols = load_pareto(args.pareto)
    sel_idx = select_solutions(sols, args.selection, args.n_runs)
    logger.info("Seleccionadas %d/%d soluciones (strategy=%s)",
                len(sel_idx), len(sols), args.selection)

    # 2. Construir Pipe (carga MLP)
    pipe = PipeWEAP(template_path=ZARR_TEMPLATE_PATH)
    target_names_gw   = pipe.surrogate.target_names_gw
    target_names_surf = pipe.surrogate.target_names_surf

    # 3. Re-evaluar cada solución y recolectar series
    decision_start = WARMUP_WEEKS + SPIN_UP_YEARS * WEEKS_PER_YEAR
    gw_series       = {}
    unmet_series    = {}
    agri_series     = {}
    metadata        = {}

    for i, sidx in enumerate(sel_idx):
        vars_, objs = sols[sidx]
        logger.info("(%d/%d) Evaluando sol#%d ...", i + 1, len(sel_idx), sidx)
        out = evaluate_solution(pipe, vars_)
        gw_series[sidx]    = extract_gw_storage_total(out["gw_denorm"], target_names_gw)
        unmet_series[sidx] = extract_unmet_ap_total(out["surf_denorm"], target_names_surf)
        agri_series[sidx]  = extract_agri_production_annual(out["surf_denorm"], target_names_surf)

        # Metadata con objetivos en unidades display (igual que plot_pareto)
        J1, J2, J3, J4, J5 = objs
        metadata[sidx] = {
            "J1_Mm3":  -J1 / 1e6,
            "J2_lps":   J2 / (ANALYSIS_HORIZON_Y * 52 * 604800) * 1000.0,
            "J3_MUSD": -J3 / USD_CLP_RATE / 1e6,
            "J4_MUSD":  J4 / USD_CLP_RATE / 1e6,
            "J5":       J5,
        }

    # 4. Plotear cada variable (estilo similar a visualize_results.py del modelo)
    common_kwargs = dict(
        decision_start_week=decision_start,
        warmup_end_week=WARMUP_WEEKS,
    )

    if any(v is not None for v in gw_series.values()):
        plot_series_overlay(
            {k: v for k, v in gw_series.items() if v is not None},
            metadata,
            title=f"TOTAL_GW_storage_Sum_AllSHACs — {len(sel_idx)} sols Pareto",
            ylabel="Storage [m³]",
            output_path=args.output_dir / "gw_storage_total.png",
            **common_kwargs,
        )

    if any(v is not None for v in unmet_series.values()):
        plot_series_overlay(
            {k: v for k, v in unmet_series.items() if v is not None},
            metadata,
            title=f"TOTAL_Unmet_AP_AllTowns — {len(sel_idx)} sols Pareto",
            ylabel="Unmet AP [l/s]",
            output_path=args.output_dir / "unmet_ap.png",
            **common_kwargs,
        )

    if any(v is not None for v in agri_series.values()):
        plot_series_overlay(
            {k: v for k, v in agri_series.items() if v is not None},
            metadata,
            title=f"TOTAL_AGR_AnnualCropProduction_AllAreas — {len(sel_idx)} sols Pareto",
            ylabel="Production [short ton/yr]",
            output_path=args.output_dir / "agri_production.png",
            **common_kwargs,
        )

    # 5. CSV resumen con metadata
    df = pd.DataFrame.from_dict(metadata, orient="index")
    df.index.name = "sol_idx"
    df.to_csv(args.output_dir / "selected_solutions_metadata.csv")
    logger.info("Metadata CSV: %s", args.output_dir / "selected_solutions_metadata.csv")

    logger.info("Todas las series guardadas en: %s", args.output_dir)


if __name__ == "__main__":
    main()
