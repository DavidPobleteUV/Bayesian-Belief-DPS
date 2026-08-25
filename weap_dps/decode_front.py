# -*- coding: utf-8 -*-
"""
decode_front.py — decodifica TODO el frente no dominado a carteras de acciones.

Las políticas son 200 pesos de una red, no un cronograma: para saber qué
proponen hay que simularlas y leer el `actions_history` del rollout. Se decodifica
cada política bajo los mismos contextos que se exportaron a WEAP, de modo que las
frecuencias del paper y los runs de verificación hablen del mismo objeto.

Guarda un npz con:
    F        (n_pol, n_obj)          objetivos optimizados
    hist     (n_pol, n_ctx, n_anios, n_acc)   estado de cada acción por año
    ctx      etiquetas de contexto
    acciones nombres de las acciones binarias

Uso:
    python weap_dps/decode_front.py --pareto_dir runs_weap/robust_iter1_fix2050
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weap_dps.action_translator import ACTION_NAMES_BINARY
from weap_dps.analyze_pareto import load as load_fronts
from weap_dps.config_weap import (DECISION_YEARS, SPIN_UP_YEARS,
                                  ZARR_TEMPLATE_PATH)
from weap_dps.export_pareto_runs import CONTEXTOS, _no_dominadas, plantilla_de_clima


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pareto_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("results/frente_decodificado.npz"))
    ap.add_argument("--limit", type=int, default=None,
                    help="decodificar solo las primeras N (para pruebas)")
    args = ap.parse_args()

    seeds, F, V, A, el, names, opt, diag = load_fronts(args.pareto_dir)
    Fall = np.vstack([F[s] for s in seeds])
    Vall = np.vstack([V[s] for s in seeds])
    nd = _no_dominadas(Fall)
    if args.limit:
        nd = nd[:args.limit]
    print(f"frente no dominado: {len(nd)} políticas de {len(Fall)}")

    from weap_dps.main_robust_weap import RobustPipeWEAP
    pipe = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=1.0)
    surr = pipe.surrogate
    plantillas = [(et, plantilla_de_clima(surr, pipe.feature_names,
                                          pipe.X_template, rid))
                  for rid, et in CONTEXTOS]

    n_acc = len(ACTION_NAMES_BINARY)
    hist = np.zeros((len(nd), len(plantillas), DECISION_YEARS, n_acc), dtype=np.int8)
    t0 = time.perf_counter()
    for k, i in enumerate(nd):
        pol = pipe._build_policy_from_params(Vall[i])
        for c, (et, X) in enumerate(plantillas):
            r = surr.rollout_with_policy(
                X_template=X, policy_fn=pol, n_years=DECISION_YEARS,
                action_col_idx=pipe.action_col_idx, spin_up_years=SPIN_UP_YEARS)
            H = np.asarray(r["actions_history"], dtype=float)[:, :n_acc]
            hist[k, c, :H.shape[0], :] = (H > 0).astype(np.int8)
        if (k + 1) % 40 == 0:
            dt = time.perf_counter() - t0
            print(f"  {k+1}/{len(nd)}  ({dt/(k+1):.2f} s/política, "
                  f"faltan {(len(nd)-k-1)*dt/(k+1)/60:.1f} min)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, F=Fall[nd], hist=hist,
        ctx=np.array([et for et, _ in plantillas], dtype=object),
        acciones=np.array(ACTION_NAMES_BINARY, dtype=object),
        # F trae solo los OPTIMIZADOS (analyze_pareto.load separa a proposito
        # los diagnosticos): guardar los 7 nombres desalinearia las columnas.
        objetivos=np.array(list(opt), dtype=object),
        idx_front=nd)
    print(f"\nguardado: {args.out}  ({time.perf_counter()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
