# -*- coding: utf-8 -*-
"""
plot_pareto.py — Visualizacion del frente de Pareto.

Genera:
  1. Plot matriz 5x5 con todos los pares de objetivos (J1..J5) en el upper
     triangle, histogramas en la diagonal y nada en el lower triangle.
  2. Plots individuales de los pares mas informativos:
        J1 vs J2  (storage GW vs Unmet AP)
        J3 vs J4  (valor agricola vs costo)
        J2 vs J4  (unmet vs costo - trade-off clasico)
        J2 vs J5  (unmet vs semanas en falla)
        J1 vs J4  (storage vs costo)

Acepta uno o varios .dat (multiples seeds) y los colorea por origen.

Uso:
    # Un solo frente
    python weap_dps/plot_pareto.py --inputs runs_weap/pareto_smoketest.dat

    # Varios seeds combinados (overlay coloreado)
    python weap_dps/plot_pareto.py --glob "runs_weap/pareto_iter01_seed*.dat" \
        --output_dir runs_weap/plots_iter01

    # Frente combinado
    python weap_dps/plot_pareto.py --inputs runs_weap/pareto_iter01_combined.dat
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [PLOT] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# Nombres y signos (algunos fueron negados para que NSGA minimice)
# Convencion:
#   negate=True    -> el valor en el .dat esta como -J (NSGA minimizo -J,
#                     que equivale a maximizar J). load_pareto_dat() devuelve J.
#   max_is_better  -> True si valores grandes son mejores (J1, J3).
#                     False si valores chicos son mejores (J2, J4, J5).
#                     Se usa en los plots para invertir ejes y que el cuadrante
#                     "mejor" quede siempre arriba-derecha.
OBJECTIVES = [
    {"key": "J1_storage_gw",  "label": "J1: GW Storage min [M m3]",            "negate": True,  "max_is_better": True,  "color_hint": "saddlebrown"},
    {"key": "J2_unmet_ap",    "label": "J2: Unmet AP [l/s promedio sostenido]","negate": False, "max_is_better": False, "color_hint": "crimson"},
    {"key": "J3_agri_value",  "label": "J3: Valor Agricola NPV [MUSD]",        "negate": True,  "max_is_better": True,  "color_hint": "darkgreen"},
    {"key": "J4_supply_cost", "label": "J4: Costo NPV [MUSD]",                 "negate": False, "max_is_better": False, "color_hint": "darkblue"},
    {"key": "J5_weeks_fail",  "label": "J5: Semanas en Falla (>100 m3/sem)",   "negate": False, "max_is_better": False, "color_hint": "purple"},
]

# ─── Conversiones de unidades raw → display ──────────────────────────────
# Constantes para conversion en load_pareto_dat. El raw .dat queda intacto;
# solo afecta lo que se ve en los plots.
USD_CLP_RATE_DISPLAY = 950.0       # CLP/USD
N_YEARS_DECISION     = 23          # años del horizonte de decisión
HORIZON_SECONDS      = N_YEARS_DECISION * 52 * 604800   # = ~7.24e8 s

# Filtro outlier: el MLP a veces predice valores GW degenerados (~0). Para
# visualizacion, descartamos puntos con J1 por debajo de este umbral.
J1_VISUAL_FLOOR_M3 = 1e6   # menos de 1 Mm3 = outlier

SEED_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#FFD92F", "#A65628", "#F781BF",
]

PAIRS_OF_INTEREST = [
    (0, 1),  # J1 vs J2
    (2, 3),  # J3 vs J4
    (1, 3),  # J2 vs J4
    (1, 4),  # J2 vs J5
    (0, 3),  # J1 vs J4
    (0, 2),  # J1 vs J3
]


# ----------------------------------------------------------------------------
# Carga y transformacion
# ----------------------------------------------------------------------------

def load_pareto_dat(path: Path) -> np.ndarray:
    """
    Devuelve array (N, 5) con los objetivos CONVERTIDOS a unidades de display:
      J1: m3 → M m3                  (÷ 1e6)
      J2: m3 acumulado → l/s prom.   (m3/horizon_s × 1000)
      J3: CLP total → MUSD/año prom. (÷ 23 ÷ 950 ÷ 1e6)
      J4: CLP NPV → MUSD             (÷ 950 ÷ 1e6)
      J5: count → count              (sin cambio)

    Pasos:
      1. Restaurar signos (los negados durante NSGA).
      2. Filtrar outliers en J1 (storage degenerado del MLP).
      3. Aplicar conversiones de unidades.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    arr = np.array([objs for _, objs in data["result"]], dtype=float)
    if arr.shape[1] != len(OBJECTIVES):
        logger.warning("Esperaba %d objetivos, .dat tiene %d. Skipping.",
                       len(OBJECTIVES), arr.shape[1])
        return None
    # 1. Restaurar signos
    for j, obj in enumerate(OBJECTIVES):
        if obj["negate"]:
            arr[:, j] = -arr[:, j]
    # 2. Filtrar outliers J1 (en m³ raw, antes de conversión)
    n_before = len(arr)
    arr = arr[arr[:, 0] >= J1_VISUAL_FLOOR_M3]
    n_dropped = n_before - len(arr)
    if n_dropped:
        logger.info("  %s: %d soluciones descartadas por J1 < %.0e",
                    path.name, n_dropped, J1_VISUAL_FLOOR_M3)
    # 3. Conversiones de unidades a display
    arr[:, 0] = arr[:, 0] / 1e6                                       # J1 → M m3
    arr[:, 1] = arr[:, 1] / HORIZON_SECONDS * 1000.0                  # J2 → l/s sostenido
    arr[:, 2] = arr[:, 2] / USD_CLP_RATE_DISPLAY / 1e6                # J3 → MUSD NPV
    arr[:, 3] = arr[:, 3] / USD_CLP_RATE_DISPLAY / 1e6                # J4 → MUSD NPV
    # J5 sin cambio
    return arr


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def plot_pair(ax, all_arrays: list, labels: list, i: int, j: int) -> None:
    """
    Scatter 2D con todos los seeds overlapeados. Orientacion uniforme:
    "mejor" siempre arriba-derecha (los ejes correspondientes a objetivos
    de minimizacion se invierten).
    """
    for idx, (arr, lbl) in enumerate(zip(all_arrays, labels)):
        color = SEED_PALETTE[idx % len(SEED_PALETTE)]
        ax.scatter(arr[:, i], arr[:, j],
                    s=40, alpha=0.7, edgecolor="black", linewidth=0.4,
                    color=color, label=lbl)
    ax.set_xlabel(OBJECTIVES[i]["label"], fontsize=10)
    ax.set_ylabel(OBJECTIVES[j]["label"], fontsize=10)
    ax.grid(True, alpha=0.3)
    # Notacion cientifica auto en ejes grandes
    for axis in ("x", "y"):
        ax.ticklabel_format(style="sci", axis=axis,
                             scilimits=(-3, 4), useMathText=True)
    # Orientacion uniforme: invertir eje si "min es mejor"
    if not OBJECTIVES[i]["max_is_better"]:
        ax.invert_xaxis()
    if not OBJECTIVES[j]["max_is_better"]:
        ax.invert_yaxis()
    # Marcar el cuadrante "mejor" con una flecha tenue en la esquina
    ax.annotate("mejor", xy=(0.95, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, alpha=0.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                          ec="gray", alpha=0.6))
    if len(all_arrays) > 1:
        ax.legend(fontsize=8, framealpha=0.8, loc="lower left")


def plot_matrix(all_arrays: list, labels: list, output_path: Path) -> None:
    """5x5: histograma en diagonal, scatter en upper, vacio en lower."""
    n = len(OBJECTIVES)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # diagonal: histograma combinado
                merged = np.concatenate([a[:, i] for a in all_arrays])
                ax.hist(merged, bins=20, color=OBJECTIVES[i]["color_hint"],
                        edgecolor="black", alpha=0.7)
                ax.set_title(OBJECTIVES[i]["label"], fontsize=9, fontweight="bold")
                ax.grid(True, alpha=0.3)
            elif j > i:
                plot_pair(ax, all_arrays, labels, j, i)
            else:
                ax.axis("off")
    fig.suptitle("Frente de Pareto - matriz de pares de objetivos",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_individual_pair(all_arrays: list, labels: list,
                          i: int, j: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_pair(ax, all_arrays, labels, i, j)
    Ji_short = OBJECTIVES[i]["label"].split(":")[0]
    Jj_short = OBJECTIVES[j]["label"].split(":")[0]
    ax.set_title(f"{Ji_short} vs {Jj_short}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", default=None)
    p.add_argument("--glob",   type=str,  default=None)
    p.add_argument("--output_dir", type=Path, default=Path("runs_weap/plots"))
    args = p.parse_args()

    # Resolver lista de inputs
    if args.inputs:
        paths = args.inputs
    elif args.glob:
        paths = [Path(p) for p in sorted(glob_mod.glob(args.glob))]
    else:
        raise SystemExit("Pasa --inputs <files...> o --glob <pattern>")
    if not paths:
        raise SystemExit("No se encontro ningun archivo.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar todos los frentes
    all_arrays = []
    labels = []
    for path in paths:
        arr = load_pareto_dat(path)
        if arr is not None and len(arr):
            all_arrays.append(arr)
            labels.append(path.stem)
            logger.info("Loaded %s -> %d soluciones", path.name, len(arr))

    if not all_arrays:
        raise SystemExit("Ningun archivo tiene soluciones validas.")

    total = sum(len(a) for a in all_arrays)
    logger.info("Total soluciones: %d  (%d frente(s))", total, len(all_arrays))

    # 1. Matriz 5x5
    matrix_out = args.output_dir / "pareto_matrix.png"
    plot_matrix(all_arrays, labels, matrix_out)

    # 2. Pares individuales
    for i, j in PAIRS_OF_INTEREST:
        Ji = OBJECTIVES[i]["label"].split(":")[0]
        Jj = OBJECTIVES[j]["label"].split(":")[0]
        fname = f"pareto_{Ji}_vs_{Jj}.png".replace(" ", "_")
        plot_individual_pair(all_arrays, labels, i, j, args.output_dir / fname)

    # 3. CSV con todas las soluciones (para inspeccion en Excel)
    csv_out = args.output_dir / "pareto_all_solutions.csv"
    with open(csv_out, "w", encoding="utf-8") as f:
        header = ["source"] + [o["key"] for o in OBJECTIVES]
        f.write(",".join(header) + "\n")
        for arr, lbl in zip(all_arrays, labels):
            for row in arr:
                f.write(",".join([lbl] + [f"{v:.6g}" for v in row]) + "\n")
    logger.info("Saved: %s", csv_out)

    logger.info("Todos los plots en: %s", args.output_dir)


if __name__ == "__main__":
    main()
