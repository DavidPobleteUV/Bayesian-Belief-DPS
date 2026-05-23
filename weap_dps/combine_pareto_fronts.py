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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", default=None,
                   help="Lista explícita de archivos .dat a combinar")
    p.add_argument("--glob",   type=str, default=None,
                   help="Patrón glob (en lugar de --inputs)")
    p.add_argument("--output", type=Path, required=True)
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
    logger.info("Frente combinado guardado: %s", args.output)


if __name__ == "__main__":
    main()
