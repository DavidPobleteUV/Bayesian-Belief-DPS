# -*- coding: utf-8 -*-
"""
compare_mlp_vs_weap.py — Compara las salidas predichas por el MLP vs las
simuladas por WEAP para los runs derivados de una iteración del exporter
de Pareto. Sirve para medir convergencia del ciclo de active learning.

Uso:
    python weap_dps/compare_mlp_vs_weap.py \
        --iteration 1 \
        --weap_zarr ../WEAP_2_ZARR/results/training_data/merged/weap_weekly.zarr \
        --output_dir data_weap/exports/iter_01/comparison

Genera:
  - divergence_per_objective.csv
  - divergence_per_variable.csv
  - plots de paridad (MLP vs WEAP) por objetivo
  - resumen.txt con statistics agregadas
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import DATA_DIR, WARMUP_WEEKS, SPIN_UP_YEARS, WEEKS_PER_YEAR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [COMPARE] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _kge(obs, sim):
    """Kling-Gupta Efficiency."""
    obs, sim = np.asarray(obs).flatten(), np.asarray(sim).flatten()
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 3:
        return np.nan
    obs, sim = obs[valid], sim[valid]
    r = np.corrcoef(obs, sim)[0, 1] if obs.std() > 0 and sim.std() > 0 else 0.0
    alpha = sim.std() / max(obs.std(), 1e-9)
    beta = sim.mean() / max(obs.mean(), 1e-9)
    return float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))


def _nse(obs, sim):
    obs, sim = np.asarray(obs).flatten(), np.asarray(sim).flatten()
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 3:
        return np.nan
    obs, sim = obs[valid], sim[valid]
    num = ((sim - obs) ** 2).sum()
    den = ((obs - obs.mean()) ** 2).sum()
    return float(1 - num / max(den, 1e-9))


def _rmse(obs, sim):
    obs, sim = np.asarray(obs).flatten(), np.asarray(sim).flatten()
    valid = np.isfinite(obs) & np.isfinite(sim)
    return float(np.sqrt(np.mean((obs[valid] - sim[valid]) ** 2))) if valid.any() else np.nan


def _pbias(obs, sim):
    obs, sim = np.asarray(obs).flatten(), np.asarray(sim).flatten()
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 3:
        return np.nan
    return float(100 * (sim[valid].sum() - obs[valid].sum()) / max(obs[valid].sum(), 1e-9))


def load_iteration_runs(iteration_dir: Path) -> pd.DataFrame:
    """Lee el master CSV de una iteración."""
    csv_path = iteration_dir / f"RunIDs_Q_pareto_iter{int(iteration_dir.name.split('_')[1]):02d}.csv"
    if not csv_path.exists():
        candidates = list(iteration_dir.glob("RunIDs_Q_pareto_iter*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Master CSV no encontrado en {iteration_dir}")
        csv_path = candidates[0]
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    logger.info("Cargado master CSV: %d runs (%s)", len(df), csv_path.name)
    return df


def find_runs_in_zarr(zarr_path: Path, run_ids: list[int]) -> dict[int, int]:
    """Devuelve {run_id: slot_index} para los run_ids encontrados en el zarr."""
    z = zarr.open_group(str(zarr_path), mode="r")
    arr = z["run_ids"][:]
    mapping = {}
    for rid in run_ids:
        idx = np.where(arr == rid)[0]
        if len(idx):
            mapping[rid] = int(idx[0])
    if not mapping:
        logger.warning("Ningún run_id de la iteración está en el zarr. "
                       "¿Ya se simuló en WEAP?")
    return mapping


def extract_weap_outputs(zarr_path: Path, slot: int) -> tuple[np.ndarray, list[str]]:
    """Devuelve (Y, target_names) para un slot dado."""
    z = zarr.open_group(str(zarr_path), mode="r")
    Y = z["Y"][slot]
    target_names = list(z.attrs.get("target_names", []))
    return Y, target_names


def compare_one_run(
    run_id: int,
    role: str,
    weap_Y: np.ndarray,
    target_names: list[str],
    mlp_Y: np.ndarray,
) -> dict:
    """Compara WEAP vs MLP para un único run. Retorna métricas agregadas."""
    # Solo desde fin de spin-up
    t0 = WARMUP_WEEKS + SPIN_UP_YEARS * WEEKS_PER_YEAR
    weap_slice = weap_Y[t0:]
    mlp_slice  = mlp_Y[t0:]
    # Alinear shapes (MLP solo predice un subconjunto de los targets WEAP)
    n_cols = min(weap_slice.shape[1], mlp_slice.shape[1])
    weap_slice = weap_slice[:, :n_cols]
    mlp_slice  = mlp_slice[:, :n_cols]

    metrics = {
        "run_id": run_id,
        "role":   role,
        "kge":    _kge(weap_slice, mlp_slice),
        "nse":    _nse(weap_slice, mlp_slice),
        "rmse":   _rmse(weap_slice, mlp_slice),
        "pbias":  _pbias(weap_slice, mlp_slice),
    }
    return metrics


def plot_parity(df_metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, metric in zip(axes.flat, ["kge", "nse", "rmse", "pbias"]):
        sub = df_metrics.dropna(subset=[metric])
        if sub.empty:
            ax.set_visible(False); continue
        ax.scatter(sub.index, sub[metric], c="steelblue", s=50, alpha=0.8)
        for _, row in sub.iterrows():
            ax.annotate(row["role"], (row.name, row[metric]),
                        fontsize=7, alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(metric.upper()); ax.grid(alpha=0.3)
        ax.set_xlabel("Run index"); ax.set_ylabel(metric)
    fig.suptitle("MLP vs WEAP convergence per run", fontweight="bold")
    fig.tight_layout()
    out = output_dir / "divergence_summary.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration",  type=int, required=True)
    p.add_argument("--exports_dir", type=Path, default=DATA_DIR / "exports")
    p.add_argument("--weap_zarr",  type=Path, required=True,
                   help="Path al zarr merged actualizado con los nuevos runs")
    p.add_argument("--output_dir", type=Path, default=None)
    args = p.parse_args()

    iter_dir = args.exports_dir / f"iter_{args.iteration:02d}"
    out_dir = args.output_dir or (iter_dir / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar master CSV
    df_master = load_iteration_runs(iter_dir)
    run_ids = df_master["ID"].astype(int).tolist()

    # 2. Encontrar runs en zarr
    slot_map = find_runs_in_zarr(args.weap_zarr, run_ids)
    if not slot_map:
        logger.error("Aborting: ningún run de la iteración se encuentra en WEAP zarr.")
        return

    # 3. Cargar MLP y re-evaluar (placeholder: dependerá del template usado)
    # Por ahora cargamos directamente las predicciones MLP de un archivo si existe;
    # si no, las re-generamos llamando al surrogate.
    mlp_preds_path = iter_dir / "mlp_predictions.npz"
    if mlp_preds_path.exists():
        mlp_preds = np.load(mlp_preds_path, allow_pickle=True)
        mlp_Y_by_run = {int(k): v for k, v in mlp_preds.items()}
        logger.info("MLP preds cacheadas en %s", mlp_preds_path)
    else:
        logger.warning("No hay cache de predicciones MLP. Tendrás que regenerar "
                        "los rollouts; saltando ese paso por ahora.")
        mlp_Y_by_run = {}

    # 4. Comparar run por run
    metrics_rows = []
    for _, row in df_master.iterrows():
        rid = int(row["ID"])
        if rid not in slot_map:
            continue
        slot = slot_map[rid]
        weap_Y, target_names = extract_weap_outputs(args.weap_zarr, slot)
        mlp_Y = mlp_Y_by_run.get(rid)
        if mlp_Y is None:
            continue
        m = compare_one_run(rid, row.get("pareto_role", ""),
                             weap_Y, target_names, mlp_Y)
        metrics_rows.append(m)

    if not metrics_rows:
        logger.warning("No se generó ninguna comparación.")
        return

    df_m = pd.DataFrame(metrics_rows)
    df_m.to_csv(out_dir / "divergence_per_run.csv", index=False)
    logger.info("Saved: %s", out_dir / "divergence_per_run.csv")

    # 5. Resumen agregado
    summary = {
        "iteration": args.iteration,
        "n_runs_compared": len(df_m),
        "kge_mean":   float(df_m["kge"].mean()),
        "kge_median": float(df_m["kge"].median()),
        "nse_mean":   float(df_m["nse"].mean()),
        "nse_median": float(df_m["nse"].median()),
        "rmse_mean":  float(df_m["rmse"].mean()),
        "pbias_mean": float(df_m["pbias"].mean()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary:")
    for k, v in summary.items():
        logger.info("  %s = %s", k, v)

    plot_parity(df_m, out_dir)
    logger.info("Comparación completada en %s", out_dir)


if __name__ == "__main__":
    main()
