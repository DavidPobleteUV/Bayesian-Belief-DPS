# -*- coding: utf-8 -*-
"""
combine_pareto_fronts.py — Combina N frentes de Pareto independientes
(distintos seeds) en un único frente no-dominado.

Para active learning con máxima diversidad: cada seed explora regiones
distintas del espacio de políticas. Al combinarlos y filtrar dominación,
el frente resultante captura la mejor solución encontrada en CUALQUIER
seed para cada zona del trade-off.

Uso:
    python weap_dps/combine_pareto_fronts.py \
        --inputs runs_weap/pareto_iter01_seed42.dat \
                 runs_weap/pareto_iter01_seed123.dat \
                 runs_weap/pareto_iter01_seed456.dat \
        --output runs_weap/pareto_iter01_combined.dat

O con glob:
    python weap_dps/combine_pareto_fronts.py \
        --glob "runs_weap/pareto_iter01_seed*.dat" \
        --output runs_weap/pareto_iter01_combined.dat
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [COMBINE] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def is_dominated(point: np.ndarray, others: np.ndarray) -> bool:
    """¿point está dominado por alguna fila de others? (minimización en todos los J)"""
    if len(others) == 0:
        return False
    le = (others <= point).all(axis=1)
    lt = (others <  point).any(axis=1)
    return bool((le & lt).any())


def filter_pareto(objectives: np.ndarray) -> np.ndarray:
    """Devuelve los índices del subconjunto no-dominado."""
    n = len(objectives)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        others = objectives[np.arange(n) != i]
        if is_dominated(objectives[i], others):
            keep[i] = False
    return np.where(keep)[0]


def load_dat(path: Path) -> list[tuple]:
    """Lee un .dat con formato {'result': [(vars, objs), ...], ...}."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["result"], data


def epsilon_grid_thin(objectives: np.ndarray, eps_frac: float) -> np.ndarray:
    """ε-dominancia (grid thinning): divide el espacio de objetivos en una
    grilla de celdas de tamaño eps_i = eps_frac * rango_i y conserva UNA
    solución por celda ocupada (la más cercana a la esquina ideal de su celda).

    Reduce el frente manteniendo cobertura uniforme del trade-off. Todos los
    objetivos están en minimización (J1/J3 ya vienen negados).

    Devuelve los índices (sobre `objectives`) de las soluciones conservadas.
    """
    n, m = objectives.shape
    mins = objectives.min(axis=0)
    rng = objectives.max(axis=0) - mins
    rng[rng <= 0] = 1.0                      # objetivos constantes → evita /0
    eps = eps_frac * rng                     # tamaño de celda por objetivo
    cells = np.floor((objectives - mins) / eps).astype(np.int64)  # (n, m)

    # distancia normalizada a la esquina inferior (ideal) de cada celda
    corner = mins + cells * eps
    dist = np.sum((objectives - corner) / eps, axis=1)

    best_per_cell: dict[tuple, tuple[float, int]] = {}
    for i in range(n):
        key = tuple(cells[i].tolist())
        if key not in best_per_cell or dist[i] < best_per_cell[key][0]:
            best_per_cell[key] = (dist[i], i)
    keep = sorted(idx for _, idx in best_per_cell.values())
    return np.array(keep, dtype=int)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", default=None,
                   help="Lista explícita de archivos .dat a combinar")
    p.add_argument("--glob",   type=str, default=None,
                   help="Patrón glob (en lugar de --inputs)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epsilon_frac", type=float, default=None,
                   help="Si se da, aplica ε-dominancia (grid thinning) con "
                        "celdas = frac × rango de cada objetivo (ej. 0.02 = 2%%). "
                        "Guarda un frente reducido en <output>_eps.dat")
    args = p.parse_args()

    # Resolver lista de inputs
    if args.inputs:
        paths = args.inputs
    elif args.glob:
        paths = [Path(p) for p in glob_mod.glob(args.glob)]
    else:
        raise SystemExit("Pasa --inputs o --glob")

    if not paths:
        raise SystemExit("No se encontró ningún input.")

    logger.info("Combinando %d frentes:", len(paths))
    for p_ in paths:
        logger.info("  %s", p_)

    # Concatenar todas las soluciones de todos los frentes
    all_vars = []
    all_objs = []
    all_sources = []
    for p_ in paths:
        results, _ = load_dat(p_)
        for vars_, objs in results:
            all_vars.append(list(vars_))
            all_objs.append(list(objs))
            all_sources.append(p_.stem)

    objs_arr = np.array(all_objs, dtype=float)
    logger.info("Soluciones agregadas (antes de filtrar dominación): %d", len(objs_arr))

    # Filtrar dominación
    keep_idx = filter_pareto(objs_arr)
    logger.info("Soluciones en frente combinado (no-dominadas): %d", len(keep_idx))

    # Estadísticas
    sources_kept = [all_sources[i] for i in keep_idx]
    from collections import Counter
    counter = Counter(sources_kept)
    logger.info("Contribución por seed:")
    for src, n in counter.most_common():
        logger.info("  %-50s %4d soluciones", src, n)

    # Guardar como .dat con formato compatible con pareto_to_runids.py
    result = [(all_vars[i], all_objs[i]) for i in keep_idx]
    out = {
        "result":  result,
        "config":  {"combined_from": [str(p) for p in paths]},
        "elapsed": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    logger.info("Frente combinado guardado: %s  (%d soluciones)", args.output, len(result))

    # ── ε-dominancia opcional: frente reducido ───────────────────────────────
    if args.epsilon_frac is not None:
        kept_objs = objs_arr[keep_idx]
        eps_idx_local = epsilon_grid_thin(kept_objs, args.epsilon_frac)
        eps_result = [result[i] for i in eps_idx_local]
        eps_out_path = args.output.with_name(args.output.stem + "_eps.dat")
        with open(eps_out_path, "wb") as f:
            pickle.dump({
                "result":  eps_result,
                "config":  {"combined_from": [str(p) for p in paths],
                            "epsilon_frac": args.epsilon_frac},
                "elapsed": None,
            }, f)
        logger.info("ε-dominancia (frac=%.3f): %d → %d soluciones representativas",
                    args.epsilon_frac, len(result), len(eps_result))
        logger.info("Frente reducido guardado: %s", eps_out_path)


if __name__ == "__main__":
    main()
