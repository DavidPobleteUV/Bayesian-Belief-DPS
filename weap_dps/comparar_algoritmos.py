# -*- coding: utf-8 -*-
"""
comparar_algoritmos.py — NSGA-II contra eps-NSGA-II sobre el mismo problema.

Responde dos preguntas distintas que conviene no mezclar:

  1. CONVERGENCIA. El hipervolumen contra el numero de evaluaciones. Si la curva
     sigue subiendo al agotarse el presupuesto, el presupuesto fue corto. La
     coincidencia del HV entre semillas NO responde esto: acota la varianza
     entre semillas, no el sesgo comun a todas.

  2. ALGORITMO. HV final, tamano del frente y fraccion no dominada dentro de
     cada semilla. Esa ultima cifra es el diagnostico de presion de seleccion:
     si vale 1.0, el rango de Pareto no esta discriminando nada y lo unico que
     empuja es la distancia de apinamiento.

Acepta .dat finales y .ckpt de corridas en curso, de modo que se puede mirar
el avance sin detener nada y abortar cuando la curva HV se aplane.

NO carga el emulador ni necesita platypus: lee los .dat y opera sobre los
objetivos guardados. Se puede correr en cualquier maquina con numpy.

La caja del hipervolumen (config.HV_MINIMUM/HV_MAXIMUM), el numero de muestras y
la semilla de Monte Carlo son FIJOS, y eso es lo unico que hace comparables los
numeros de corridas distintas. Ver hv_utils para por que la estimacion en vez
del valor exacto.

Uso:
    python weap_dps/comparar_algoritmos.py \
        --nsga "results/iter02_pareto_seed*.dat" \
        --eps  "runs_weap/eps/pareto_seed*.dat"
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weap_dps.config_weap import (  # noqa: E402
    HV_MAXIMUM, HV_MINIMUM, OBJECTIVES_OPTIMIZED,
)
from weap_dps.hv_utils import hipervolumen_mc  # noqa: E402


def hv(objetivos: np.ndarray) -> tuple[float, int]:
    """HV sobre la caja fija, y cuantas soluciones caen fuera de ella."""
    return hipervolumen_mc(objetivos, HV_MINIMUM, HV_MAXIMUM)


def frac_no_dominada(A: np.ndarray) -> float:
    """Fraccion del conjunto que es no dominada DENTRO de si mismo.

    En 1.0 la presion de seleccion por rango de Pareto es exactamente nula: no
    hay ningun par comparable y solo la diversidad decide quien sobrevive.
    """
    n = len(A)
    if n == 0:
        return float("nan")
    dom = np.zeros(n, bool)
    for i in range(n):
        # j domina a i si es <= en todo y < en algo
        mejor_o_igual = np.all(A <= A[i], axis=1)
        estrictamente = np.any(A < A[i], axis=1)
        dom[i] = bool(np.any(mejor_o_igual & estrictamente))
    return float((~dom).mean())


def cargar(patron: str) -> list[dict]:
    """Lee .dat finales y tambien .ckpt de corridas en curso o abortadas.

    Poder leer el checkpoint importa por una razon practica: con el HV
    registrado se puede abortar una corrida cuando la curva se aplana, y sin
    esto ese aborto no dejaria nada utilizable, porque el .dat solo se escribe
    al terminar. El .ckpt guarda el archivo eps y la historia de HV, que es
    todo lo que necesita esta comparacion.
    """
    out = []
    for f in sorted(glob.glob(patron)):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if "result" in d:                       # .dat final
            A = np.array([o for _, o in d["result"]], float)
        elif "archive" in d:                    # .ckpt de una corrida en curso
            A = np.array([o for _, o in d["archive"]], float)
            d = dict(d, nfe_real=d.get("nfe"), elapsed=float("nan"))
        else:
            print(f"  (se omite {Path(f).name}: no trae frente ni archivo)")
            continue
        out.append({"archivo": Path(f).name, "obj": A, "dat": d})
    return out


def resumen(nombre: str, corridas: list[dict]) -> dict | None:
    if not corridas:
        print(f"\n{nombre}: sin archivos")
        return None
    print(f"\n{'='*78}\n{nombre}  ({len(corridas)} semillas)\n{'='*78}")
    print(f"{'archivo':<34}{'nfe':>7}{'frente':>8}{'HV':>10}"
          f"{'no dom.':>10}{'horas':>8}{'fuera':>7}")
    hvs, fuera_total = [], 0
    for c in corridas:
        h, fu = hv(c["obj"])
        hvs.append(h)
        fuera_total += fu
        horas = c["dat"].get("elapsed", float("nan")) / 3600
        # nfe REAL: eps-NSGA-II puede exceder lo pedido porque la descendencia
        # de un reinicio se evalua en bloque. Comparar dos corridas por su HV
        # final sin mirar esta columna seria comparar presupuestos distintos.
        cfg = c["dat"].get("config", {})
        nfe = c["dat"].get("nfe_real", cfg.get("evaluations", float("nan")))
        print(f"{c['archivo']:<34}{nfe:>7.0f}{len(c['obj']):>8}{h:>10.5f}"
              f"{frac_no_dominada(c['obj']):>10.2f}{horas:>8.1f}{fu:>7d}")
    union = np.vstack([c["obj"] for c in corridas])
    h_union, fu_union = hv(union)
    hvs = np.array(hvs)
    cv = 100 * hvs.std() / hvs.mean() if hvs.mean() else float("nan")
    print(f"\n  HV medio        {hvs.mean():.5f}   CV entre semillas {cv:.1f} %")
    print(f"  HV de la union  {h_union:.5f}   ({len(union)} soluciones, "
          f"{frac_no_dominada(union):.2f} no dominadas)")
    if fuera_total:
        print(f"  AVISO: {fuera_total} soluciones fuera de la caja de HV. Se "
              f"recortan al nadir, asi que el HV de esa corrida queda sesgado "
              f"al alza. Amplia HV_MAXIMUM en config_weap.py y recalcula todo.")
    return {"hv_medio": float(hvs.mean()), "cv": float(cv), "hv_union": h_union}


def curva_hv(corridas: list[dict]) -> None:
    """Trayectoria HV(nfe) de las corridas que la hayan registrado."""
    con = [c for c in corridas if c["dat"].get("hv_history")]
    if not con:
        return
    print(f"\n{'='*78}\nConvergencia: HV contra evaluaciones\n{'='*78}")
    for c in con:
        h = c["dat"]["hv_history"]
        nfe = [r["nfe"] for r in h]
        vals = [r["hv"] for r in h]
        print(f"\n  {c['archivo']}")
        # Se muestran ~10 puntos repartidos, no los 50 checkpoints.
        idx = np.linspace(0, len(h) - 1, min(10, len(h))).astype(int)
        for i in idx:
            print(f"    nfe {nfe[i]:>6}  HV {vals[i]:.5f}  "
                  f"archivo {h[i]['archivo']:>4}  poblacion {h[i]['poblacion']:>4}")
        # Ganancia del ultimo cuarto del presupuesto: si es despreciable, la
        # curva se aplano y el presupuesto alcanzo. Si no, falto.
        k = max(1, len(vals) // 4)
        ini, fin = vals[-k - 1], vals[-1]
        gan = 100 * (fin - ini) / ini if ini else float("nan")
        print(f"    ganancia del ultimo cuarto del presupuesto: {gan:+.2f} %")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsga", default="results/iter02_pareto_seed*.dat")
    ap.add_argument("--eps", default="runs_weap/eps/pareto_seed*.dat")
    a = ap.parse_args()

    print("Caja de hipervolumen (fija, config_weap.py):")
    for nm, lo, hi in zip(OBJECTIVES_OPTIMIZED, HV_MINIMUM, HV_MAXIMUM):
        print(f"  {nm:<22} ideal {lo:>12.4g}   nadir {hi:>12.4g}")

    n = cargar(a.nsga)
    e = cargar(a.eps)
    rn = resumen("NSGA-II", n)
    re_ = resumen("eps-NSGA-II", e)
    curva_hv(n + e)

    if rn and re_:
        d = 100 * (re_["hv_union"] - rn["hv_union"]) / rn["hv_union"]
        print(f"\n{'='*78}")
        print(f"HV de la union: NSGA-II {rn['hv_union']:.5f}  ->  "
              f"eps-NSGA-II {re_['hv_union']:.5f}   ({d:+.1f} %)")
        print("Mayor es mejor. Una diferencia del orden del CV entre semillas "
              "no es evidencia de nada.")
        print("Si las columnas 'nfe' no coinciden, esta linea compara "
              "presupuestos distintos: lee la comparacion en la curva HV(nfe).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
