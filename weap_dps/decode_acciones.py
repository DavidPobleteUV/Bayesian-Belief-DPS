# -*- coding: utf-8 -*-
"""Que acciones activan las politicas del frente, y en que anio.

Decodifica una muestra del frente combinado simulandola: las acciones no estan
en el .dat (solo los 200 pesos de la politica), salen del actions_history del
rollout. Se usan pocos estados del mundo por politica: basta para ver QUE
acciones usa cada una, no para reevaluar objetivos.
"""
import sys, pickle, numpy as np
from pathlib import Path
sys.path.insert(0, ".")
from weap_dps.analyze_pareto import load
from weap_dps.main_robust_weap import RobustPipeWEAP
from weap_dps.config_weap import ZARR_TEMPLATE_PATH
from weap_dps.action_translator import ACTION_NAMES_BINARY
from weap_dps.config_weap import DECISION_YEARS, SPIN_UP_YEARS

D = Path("runs_weap/robust_iter1_fix2050")
N_POL = int(sys.argv[1]) if len(sys.argv) > 1 else 24
N_SOW = int(sys.argv[2]) if len(sys.argv) > 2 else 3

seeds, F, V, A, el, names, opt, diag = load(D)
Vall = np.vstack([V[s] for s in seeds])
Fall = np.vstack([F[s] for s in seeds])

# frente no dominado combinado
nd = []
for i in range(len(Fall)):
    if not any((Fall[j] <= Fall[i]).all() and (Fall[j] < Fall[i]).any()
               for j in range(len(Fall)) if j != i):
        nd.append(i)
nd = np.array(nd)
print(f"frente combinado: {len(nd)} politicas no dominadas de {len(Fall)}")
sel = nd[np.linspace(0, len(nd) - 1, min(N_POL, len(nd))).astype(int)]
print(f"muestra decodificada: {len(sel)} politicas x {N_SOW} estados del mundo\n")

sim = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=1.0)
sim.scenarios = sim.scenarios[:N_SOW]
cuenta = {a: 0 for a in ACTION_NAMES_BINARY}
anio_primero = {a: [] for a in ACTION_NAMES_BINARY}
n_acc_por_pol = []
for k, i in enumerate(sel):
    pol = sim._build_policy_from_params(Vall[i])
    hist = []
    for Xs in sim.scenarios:
        r = sim.surrogate.rollout_with_policy(
            X_template=Xs, policy_fn=pol, n_years=DECISION_YEARS,
            action_col_idx=sim.action_col_idx, spin_up_years=SPIN_UP_YEARS)
        hist.extend(r["actions_history"])
    H = np.asarray(hist, dtype=float)          # (n_anios, n_acciones)
    usadas = set()
    for j, a in enumerate(ACTION_NAMES_BINARY):
        serie = list(H[:, j])
        if max(serie) > 0:
            usadas.add(a); cuenta[a] += 1
            anio_primero[a].append(2027 + int(np.argmax(H[:, j] > 0)))
    n_acc_por_pol.append(len(usadas))
    if (k + 1) % 6 == 0:
        print(f"  {k+1}/{len(sel)}")

print(f"\n{'accion':26s} {'% del frente':>13} {'anio activacion (mediana)':>26}")
for a in ACTION_NAMES_BINARY:
    p = 100 * cuenta[a] / max(len(n_acc_por_pol), 1)
    y = int(np.median(anio_primero[a])) if anio_primero[a] else None
    print(f"{a:26s} {p:12.0f}% {str(y) if y else '-':>26}")
print(f"\nacciones por politica: mediana={np.median(n_acc_por_pol):.0f}  "
      f"min={min(n_acc_por_pol)}  max={max(n_acc_por_pol)}")
