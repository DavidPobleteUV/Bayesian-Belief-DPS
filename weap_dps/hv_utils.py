# -*- coding: utf-8 -*-
"""
hv_utils.py — Hipervolumen sobre una caja FIJA, estimado por Monte Carlo.

POR QUE NO SE USA EL DE PLATYPUS. Su implementacion es el algoritmo recursivo
exacto en Python puro, y con 5 objetivos escala de forma prohibitiva. Medido
sobre nuestro propio frente:

    n =  10 soluciones ->   0.00 s
    n =  20            ->   0.02 s
    n =  40            ->   0.54 s        (~27x por cada duplicacion)
    n = 100            ->   ~40 s
    n = 500            ->   no termino en 25 min

Como el archivo de eps-NSGA-II puede tener cientos de soluciones y el HV se
calcula muchas veces durante una corrida, el metodo exacto haria que medir la
convergencia costara mas que optimizar.

QUE SE HACE EN CAMBIO. Se estima la fraccion del volumen de la caja que queda
dominada por al menos una solucion, muestreando puntos al azar. El costo es
lineal en el numero de soluciones y de muestras.

Dos propiedades que lo hacen valido para lo que se usa:

  - Con SEMILLA Y NUMERO DE MUESTRAS FIJOS, dos conjuntos se evaluan contra los
    MISMOS puntos de muestreo. El error de estimacion queda entonces
    correlacionado entre ellos y las COMPARACIONES son mas precisas que el
    error absoluto de cada valor por separado. Es exactamente el regimen que
    interesa: comparar HV entre checkpoints, semillas y algoritmos.
  - El error estandar de una fraccion con N muestras es sqrt(p(1-p)/N), o sea
    ~0.001 con N = 200.000. Las diferencias que discutimos son de varios por
    ciento.

NO usar estos valores como un hipervolumen exacto publicable sin declarar el
metodo: son una estimacion, y asi hay que reportarla.
"""
from __future__ import annotations

import numpy as np

# Fijos a proposito: cambiarlos invalida la comparacion con los HV ya
# calculados, igual que cambiar la caja.
N_MUESTRAS = 200_000
SEMILLA_MC = 20260920


def hipervolumen_mc(objetivos, minimum, maximum,
                    n_muestras: int = N_MUESTRAS,
                    semilla: int = SEMILLA_MC) -> tuple[float, int]:
    """Fraccion de la caja dominada, y cuantas soluciones caen fuera de ella.

    Convencion de MINIMIZACION: una solucion domina un punto de muestreo si es
    menor o igual en TODOS los objetivos. `minimum` es el ideal y `maximum` el
    nadir de referencia.

    El segundo valor devuelto es el contador de soluciones fuera de la caja.
    Las que exceden el nadir se RECORTAN al nadir en vez de descartarse —
    platypus las descarta en silencio, que es peor: un frente peor pasaria a
    medir mas que uno mejor solo porque sus puntos malos desaparecen—. Pero el
    contador se devuelve igual: si deja de ser cero, la caja quedo chica y
    conviene ampliarla en config_weap.py antes de comparar nada.
    """
    A = np.asarray(objetivos, dtype=float)
    if A.size == 0:
        return 0.0, 0
    lo = np.asarray(minimum, dtype=float)
    hi = np.asarray(maximum, dtype=float)

    fuera = int(np.any((A > hi) | (A < lo), axis=1).sum())
    A = np.clip(A, lo, hi)

    # Normalizacion a [0,1]^m; el volumen de la caja es entonces 1 y el HV sale
    # directamente como fraccion, comparable entre objetivos de escalas dispares.
    U = (A - lo) / np.where(hi - lo == 0, 1.0, hi - lo)

    rng = np.random.default_rng(semilla)
    dominados = 0
    # Por bloques: la matriz completa (n_muestras x m) por cada solucion no cabe
    # comoda en memoria y tampoco hace falta tenerla entera.
    bloque = 50_000
    restantes = n_muestras
    while restantes > 0:
        k = min(bloque, restantes)
        S = rng.random((k, U.shape[1]))
        vivo = np.ones(k, dtype=bool)          # aun no dominado por ninguna
        for u in U:
            if not vivo.any():
                break
            # Minimizacion: la solucion u cubre el punto de muestreo s si u es
            # menor o igual que s en TODOS los objetivos, es decir s >= u.
            vivo[vivo] = ~np.all(S[vivo] >= u, axis=1)
        dominados += int((~vivo).sum())
        restantes -= k
    return dominados / n_muestras, fuera
