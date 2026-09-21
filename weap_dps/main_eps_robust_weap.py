# -*- coding: utf-8 -*-
"""
main_eps_robust_weap.py — Robust DPS con ε-NSGA-II.

VERSIÓN PARALELA a main_robust_weap.py, que se conserva intacto. Las dos deben
poder correrse sobre el mismo problema y compararse, así que la evaluación de
objetivos NO se duplica aquí: se importa `RobustPipeWEAP` de la versión NSGA-II.
Si se copiara, cualquier divergencia futura —una calibración, un umbral— haría
que la comparación midiera dos cosas distintas creyendo medir el algoritmo.

Qué cambia respecto de la versión NSGA-II, y por qué:

1. ARCHIVO DE ε-DOMINANCIA. Con 5 objetivos el rango de Pareto deja de
   discriminar: en iter02 las 5 semillas devolvieron un frente de 100 sobre una
   población de 100, es decir la población entera mutuamente no dominada y
   presión de selección nula al final de la corrida.

   El archivo NO actúa sobre esa presión directamente. En platypus los padres
   salen de la población y la truncación sigue las reglas de NSGA-II; el
   archivo solo recibe soluciones (archive.extend). Influye en la búsqueda por
   una única vía, los reinicios (punto 2), que reconstruyen la población desde
   él. Lo que aporta por sí mismo es un frente con resolución declarada
   —config.EPSILONS: cuánta diferencia es significativa en cada objetivo— y de
   tamaño acotado. Si además mejora la presión en la población es una pregunta
   empírica, que responde comparar_algoritmos.py a igual número de evaluaciones.

2. REINICIOS ADAPTATIVOS. ε-NSGA-II reescala la población al tamaño del archivo
   y reinyecta diversidad cuando detecta estancamiento. Los valores por defecto
   de platypus (ventana de 100 GENERACIONES, población hasta 10000) están
   pensados para presupuestos mucho mayores: con 4000-10000 evaluaciones y
   población 100 tenemos 40-100 generaciones, así que el mecanismo no llegaría
   a dispararse nunca. Se exponen como parámetros y se bajan por defecto.

3. REGISTRO DEL HIPERVOLUMEN. La versión anterior solo permitía comparar
   semillas al final, lo que acota la varianza entre semillas pero no dice si el
   presupuesto alcanzó. Aquí se guarda HV(nfe) en cada checkpoint: si la curva
   sigue subiendo al final, faltaron evaluaciones. Esa es la única forma
   empírica de responder "¿cuántas evaluaciones son suficientes?".

Uso:
    DPS_CKPT=<ckpt> python weap_dps/main_eps_robust_weap.py \
        --evaluations 10000 --population 100 --seed 42 \
        --n_climate 5 --lam 1.0 --output runs_weap/eps/pareto_seed42.dat
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# RobustPipeWEAP fija los hilos de torch al importarse (antes de importar torch),
# de modo que este import debe ir antes que cualquier otro que arrastre torch.
from weap_dps.main_robust_weap import RobustPipeWEAP  # noqa: E402

from platypus import (  # noqa: E402
    EpsNSGAII, InjectedPopulation, Solution,
)
from platypus.extensions import (  # noqa: E402
    AdaptiveTimeContinuationExtension, EpsilonProgressContinuationExtension,
)

from weap_dps.config_weap import (  # noqa: E402
    RESULTS_DIR, ZARR_TEMPLATE_PATH, EPSILONS, EPSILONS_BY_OBJECTIVE, EPS_SCALE,
    HV_MINIMUM, HV_MAXIMUM,
    OBJECTIVES_OPTIMIZED, OBJECTIVES_DIAGNOSTIC, OBJECTIVE_NAMES,
)
from weap_dps.hv_utils import hipervolumen_mc  # noqa: E402
from weap_dps.pipe_problem_weap import PipeProblemWEAP  # noqa: E402
from weap_dps.scenario_builder import build_scenarios  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [EPS] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hipervolumen sobre una caja FIJA
# ─────────────────────────────────────────────────────────────────────────────
def hipervolumen(objetivos) -> tuple[float, int]:
    """HV sobre la caja fija de config, y cuántas soluciones caen fuera de ella.

    Estimado por Monte Carlo (ver hv_utils): el hipervolumen exacto de platypus
    en 5 objetivos tarda ~40 s con 100 soluciones y no termina con 500, de modo
    que medir la convergencia costaría más que optimizar. La estimación queda a
    0.2 % del valor exacto donde éste sí es calculable, y tarda 0.6 s.
    """
    return hipervolumen_mc(objetivos, HV_MINIMUM, HV_MAXIMUM)


def _sols_desde(problem, pares) -> list[Solution]:
    """Reconstruye soluciones YA EVALUADAS desde (variables, objetivos)."""
    out = []
    for v, o in pares:
        s = Solution(problem)
        s.variables[:] = list(v)
        s.objectives[:] = list(o)
        s.evaluated = True
        out.append(s)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluations", type=int, default=10000)
    p.add_argument("--population", type=int, default=100,
                   # Sin acentos ni griegas en los help: argparse los imprime a
                   # stdout, y en Windows con cp1252 un --help revienta con
                   # UnicodeEncodeError antes de llegar a correr nada.
                   help="poblacion INICIAL; eps-NSGA-II la reescala en cada reinicio")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_climate", type=int, default=5)
    p.add_argument("--n_sow", type=int, default=None,
                   help="estados del mundo; por defecto config.DPS_N_SOW (27). "
                        "Bajarlo SOLO para pruebas de humo: cambia el problema")
    p.add_argument("--lam", type=float, default=1.0,
                   help="aversion al riesgo (mean + lambda*std)")
    p.add_argument("--output", type=Path,
                   default=RESULTS_DIR / f"eps_robust_{int(time.time())}.dat")
    p.add_argument("--checkpoint_every", type=int, default=200,
                   help="evaluaciones entre checkpoints; 0 los desactiva")
    p.add_argument("--hv_every", type=int, default=2,
                   help="calcular el hipervolumen cada N checkpoints")
    p.add_argument("--no_resume", action="store_true")
    # ── Continuación adaptativa ─────────────────────────────────────────────
    p.add_argument("--restart_window", type=int, default=10,
                   help="generaciones entre chequeos de reinicio (platypus usa 100, "
                        "inalcanzable con nuestro presupuesto)")
    p.add_argument("--max_population", type=int, default=300,
                   help="tope de poblacion tras un reinicio. Se escala a 4x el "
                        "tamano del archivo, y sin tope una generacion podria "
                        "costar mas que el presupuesto entero")
    p.add_argument("--eps_progress", action="store_true",
                   help="usa continuacion por eps-progreso (reinicia cuando el "
                        "archivo deja de mejorar, como Borg) en vez de la "
                        "continuacion temporal por defecto")
    args = p.parse_args()

    np.random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pipe = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=args.lam)
    scen, labels = build_scenarios(pipe.surrogate, pipe.feature_names,
                                   pipe.X_template, n_climate=args.n_climate,
                                   n_sow=args.n_sow)
    pipe.scenarios = scen
    logger.info("Ensamble robusto: %d escenarios (clima x demanda)  lambda=%.2f",
                len(scen), args.lam)
    logger.info("epsilons (escala %.3g):", EPS_SCALE)
    for nm, e in zip(OBJECTIVES_OPTIMIZED, EPSILONS):
        logger.info("   %-22s %.4g   (base %.4g)", nm, e, EPSILONS_BY_OBJECTIVE[nm])

    problem = PipeProblemWEAP(pipe)

    # ── Reanudación ─────────────────────────────────────────────────────────
    # Se reinyecta el ARCHIVO, no la población. Dos razones: el archivo es el
    # estado valioso —es el frente— y es además lo que el propio algoritmo usa
    # al reiniciar (restart() hace population = archive[:] + mutantes), de modo
    # que reanudar así es consistente con su semántica. Al inicializar,
    # EpsNSGAII hace archive += population, con lo que el archivo se restituye
    # solo y no hace falta restaurarlo por separado.
    ck = args.output.with_suffix(".ckpt")
    hecho, hv_hist = 0, []
    pop_size = args.population
    inject = None
    if ck.exists() and not args.no_resume:
        with open(ck, "rb") as f:
            est = pickle.load(f)
        inject = _sols_desde(problem, est["archive"])
        pipe._diag.update({tuple(k): np.asarray(v) for k, v in est["diag"]})
        hecho = int(est["nfe"])
        hv_hist = list(est.get("hv_history", []))
        # Si el archivo creció por encima de la población guardada, la población
        # debe caber: InjectedPopulation descarta el sobrante en silencio y se
        # perderían soluciones del frente.
        pop_size = max(int(est.get("population_size", args.population)), len(inject))
        logger.info("REANUDANDO desde %s: %d evaluaciones, archivo=%d, poblacion=%d",
                    ck.name, hecho, len(inject), pop_size)

    algo = EpsNSGAII(problem, epsilons=list(EPSILONS), population_size=pop_size,
                     **({"generator": InjectedPopulation(inject)} if inject else {}))

    # ── Continuación adaptativa con parámetros utilizables ──────────────────
    # EpsNSGAII añade AdaptiveTimeContinuationExtension con window_size=100
    # GENERACIONES. Con población 100 y 10000 evaluaciones hay 100 generaciones,
    # así que el chequeo ocurriría una sola vez, al final: el mecanismo estaría
    # nominalmente activo y en la práctica muerto. Se reemplaza por uno con
    # ventana corta y población acotada.
    algo.remove_extension(AdaptiveTimeContinuationExtension)
    Ext = (EpsilonProgressContinuationExtension if args.eps_progress
           else AdaptiveTimeContinuationExtension)
    algo.add_extension(Ext(window_size=args.restart_window,
                           max_window_size=10 * args.restart_window,
                           max_population_size=args.max_population))
    logger.info("continuacion: %s  ventana=%d generaciones  poblacion<=%d",
                Ext.__name__, args.restart_window, args.max_population)

    def objetivos_archivo():
        return [list(s.objectives) for s in algo.archive]

    def guardar_ck(nfe):
        tmp = ck.with_suffix(".ckpt.tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"nfe": nfe,
                         "population": [(list(x.variables), list(x.objectives))
                                        for x in algo.population],
                         "archive": [(list(x.variables), list(x.objectives))
                                     for x in algo.archive],
                         "diag": [(list(k), v.tolist()) for k, v in pipe._diag.items()],
                         "hv_history": hv_hist,
                         "population_size": len(algo.population),
                         "seed": args.seed}, f)
        tmp.replace(ck)          # atómico: un corte durante el volcado no corrompe

    t0 = time.time()
    paso = args.checkpoint_every if args.checkpoint_every > 0 else args.evaluations
    # Misma contabilidad que en la versión NSGA-II: `algo.nfe` cuenta desde cero
    # en cada instancia, e inicializar con InjectedPopulation consume nfe SIN
    # evaluar (platypus incrementa nfe por len(solutions), estén evaluadas o no).
    # Sin el offset, una reanudación se daría por completa sin hacer trabajo.
    #
    # EL PRESUPUESTO NO ES EXACTO, Y HAY QUE SABERLO. Un reinicio evalúa su
    # descendencia en bloque —restart() llama a evaluate_all(offspring)— dentro
    # de un paso, así que el paso puede sobrepasar lo pedido por hasta
    # (max_population - tamaño del archivo) evaluaciones. En la prueba de humo,
    # con ventana de 1 generación, 60 pedidas terminaron en 115.
    #
    # No se "arregla" recortando el reinicio, porque mutilarlo cambiaría el
    # algoritmo que se quiere medir. Lo que se hace es REGISTRAR el gasto real y
    # avisar del exceso, y comparar contra NSGA-II por la curva HV(nfe) —que
    # permite leer ambos al MISMO número de evaluaciones— en vez de por el valor
    # final de cada .dat, que estaría medido con presupuestos distintos.
    base = hecho
    offset = min(len(inject), pop_size) if inject else 0
    objetivo = (args.evaluations - base) + offset
    vuelta = 0
    while algo.nfe < objetivo:
        algo.run(min(paso, objetivo - algo.nfe))
        hecho = base + algo.nfe - offset
        vuelta += 1
        # El HV estimado cuesta <1 s, despreciable frente a 58.5 s por
        # evaluacion, pero igual no hay razon para pagarlo en cada checkpoint:
        # el checkpoint protege la corrida (barato) y la curva HV solo necesita
        # resolucion suficiente para ver si se aplano.
        ultimo = algo.nfe >= objetivo
        if vuelta % max(1, args.hv_every) == 0 or ultimo:
            hv, fuera = hipervolumen(objetivos_archivo())
            hv_hist.append({"nfe": hecho, "hv": hv, "fuera_de_caja": fuera,
                            "archivo": len(algo.archive),
                            "poblacion": len(algo.population),
                            "minutos": (time.time() - t0) / 60})
            logger.info("%d/%d evaluaciones | HV=%.5f | archivo=%d | poblacion=%d | %.1f min%s",
                        hecho, args.evaluations, hv, len(algo.archive),
                        len(algo.population), (time.time() - t0) / 60,
                        "" if fuera == 0 else f" | AVISO: {fuera} fuera de la caja de HV")
        else:
            logger.info("%d/%d evaluaciones | archivo=%d | poblacion=%d | %.1f min",
                        hecho, args.evaluations, len(algo.archive),
                        len(algo.population), (time.time() - t0) / 60)
        if args.checkpoint_every > 0:
            guardar_ck(hecho)

    el = time.time() - t0
    exceso = hecho - args.evaluations
    logger.info("Listo en %.1f min | archivo=%d | evaluaciones reales %d "
                "(pedidas %d, exceso %+d = %+.1f %% por descendencia de reinicios)",
                el / 60, len(algo.archive), hecho, args.evaluations, exceso,
                100 * exceso / args.evaluations)

    n_hit = sum(pipe.all_objectives_for(s.variables) is not None for s in algo.archive)
    logger.info("Diagnostico (J1, J6) recuperado para %d/%d politicas del archivo",
                n_hit, len(algo.archive))

    if ck.exists():
        ck.unlink()

    hv, fuera = hipervolumen(objetivos_archivo())
    with open(args.output, "wb") as f:
        pickle.dump({"result": [(list(s.variables), list(s.objectives))
                                for s in algo.archive],
                     "all_objectives": [
                         (lambda v: None if v is None else list(v))(
                             pipe.all_objectives_for(s.variables))
                         for s in algo.archive],
                     "objective_names": OBJECTIVE_NAMES,
                     "objectives_optimized": OBJECTIVES_OPTIMIZED,
                     "objectives_diagnostic": OBJECTIVES_DIAGNOSTIC,
                     "algorithm": "EpsNSGAII",
                     "epsilons": list(EPSILONS),
                     "eps_scale": EPS_SCALE,
                     "hv_final": hv, "hv_fuera_de_caja": fuera,
                     # Evaluaciones REALES, que pueden exceder las pedidas por
                     # la descendencia de los reinicios. Cualquier comparacion
                     # de presupuesto debe usar este numero, no --evaluations.
                     "nfe_real": hecho,
                     "nfe_pedidas": args.evaluations,
                     "hv_box": {"minimum": list(HV_MINIMUM), "maximum": list(HV_MAXIMUM)},
                     "hv_history": hv_hist,
                     # La POBLACIÓN final, para el diagnóstico de presión de
                     # selección. El frente ("result") es el archivo eps, cuya
                     # fracción no dominada vale 1.0 por construcción.
                     "population": [(list(s.variables), list(s.objectives))
                                    for s in algo.population],
                     "scenarios": labels, "lam": args.lam,
                     "config": vars(args), "elapsed": el}, f)
    logger.info("Guardado: %s  (HV final %.5f)", args.output, hv)


if __name__ == "__main__":
    main()
