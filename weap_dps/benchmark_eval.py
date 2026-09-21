# -*- coding: utf-8 -*-
"""
benchmark_eval.py — Costo real de una evaluacion en ESTA maquina.

El ETA de los lanzadores usa 2.17 s por escenario, MEDIDO en la PC de trabajo
(65.0 h / 4000 evaluaciones / 27 escenarios). Extrapolarlo a otra maquina es
adivinar: lo que manda es la velocidad de UN hilo, no el numero de cores, porque
cada semilla corre con OMP_NUM_THREADS=1.

Esto mide el costo real en unos minutos y proyecta el reloj de la corrida
completa, para decidir el presupuesto ANTES de comprometer dias de maquina.

Tambien mide con varios procesos en paralelo, que es lo que interesa de verdad:
si la memoria compartida es el cuello de botella, N semillas en paralelo son
mas lentas por semilla que una sola, y el ETA calculado con una sola miente.

Uso:
    python weap_dps/benchmark_eval.py                 # 3 evaluaciones, 1 proceso
    python weap_dps/benchmark_eval.py --n 5 --par 8   # y ademas 8 en paralelo
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weap_dps.main_robust_weap import RobustPipeWEAP  # noqa: E402  (fija hilos de torch)
from weap_dps.config_weap import ZARR_TEMPLATE_PATH, DPS_N_SOW  # noqa: E402
from weap_dps.pipe_problem_weap import PipeProblemWEAP  # noqa: E402
from weap_dps.scenario_builder import build_scenarios  # noqa: E402


def medir(n_eval: int, n_sow: int | None, seed: int = 0) -> tuple[float, int]:
    """Segundos por evaluacion, promediando n_eval politicas al azar."""
    rng = np.random.default_rng(seed)
    pipe = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=1.0)
    scen, _ = build_scenarios(pipe.surrogate, pipe.feature_names,
                              pipe.X_template, n_climate=5, n_sow=n_sow)
    pipe.scenarios = scen
    problem = PipeProblemWEAP(pipe)
    n_var = problem.nvars

    # Una evaluacion en frio para no medir la carga perezosa de torch.
    pipe.simulation(rng.uniform(-3, 3, n_var))

    t0 = time.time()
    for _ in range(n_eval):
        pipe.simulation(rng.uniform(-3, 3, n_var))
    return (time.time() - t0) / n_eval, len(scen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="evaluaciones a promediar")
    ap.add_argument("--n_sow", type=int, default=None,
                    help="estados del mundo; por defecto config.DPS_N_SOW")
    ap.add_argument("--par", type=int, default=0,
                    help="ademas, medir con N procesos en paralelo (0 = no)")
    ap.add_argument("--presupuestos", type=int, nargs="*",
                    default=[4000, 10000, 20000],
                    help="presupuestos por semilla para proyectar el reloj")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._worker:
        s, _ = medir(a.n, a.n_sow, seed=os.getpid())
        print(f"{s:.4f}")
        return 0

    import multiprocessing
    print(f"maquina: {multiprocessing.cpu_count()} cores logicos  |  "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(sin fijar)')}")
    print(f"escenarios por evaluacion: {a.n_sow or DPS_N_SOW}\n")

    s1, n_scen = medir(a.n, a.n_sow)
    print(f"1 proceso  : {s1:7.2f} s/evaluacion   ({s1/n_scen:.3f} s/escenario)")

    s_par = None
    if a.par > 0:
        # Procesos de verdad, no hilos: es la forma en que corren las semillas y
        # es la unica que expone la competencia por memoria y por cache.
        cmd = [sys.executable, __file__, "--_worker", "--n", str(a.n)]
        if a.n_sow:
            cmd += ["--n_sow", str(a.n_sow)]
        t0 = time.time()
        procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
                 for _ in range(a.par)]
        vals = []
        for p in procs:
            out, _ = p.communicate()
            try:
                vals.append(float(out.strip().splitlines()[-1]))
            except (ValueError, IndexError):
                pass
        if vals:
            s_par = float(np.mean(vals))
            print(f"{a.par} procesos: {s_par:7.2f} s/evaluacion   "
                  f"(penalizacion {100*(s_par/s1-1):+.0f} % por proceso, "
                  f"medido en {time.time()-t0:.0f} s)")
        else:
            print(f"{a.par} procesos: fallo la medicion en paralelo")

    s = s_par or s1
    print(f"\nProyeccion del reloj con {s:.2f} s/evaluacion "
          f"(semillas en paralelo, el reloj lo fija el presupuesto POR SEMILLA):")
    for e in a.presupuestos:
        print(f"  {e:>6} evaluaciones -> {e*s/3600:6.1f} h  ({e*s/86400:.1f} dias)")
    if s_par is None and a.par == 0:
        print("\n  OJO: medido con 1 solo proceso. Con varias semillas en "
              "paralelo puede ser mas lento; repite con --par N.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
