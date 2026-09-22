# -*- coding: utf-8 -*-
"""
estados_disparo.py — qué observaba la política cuando activó cada acción.

Nivel 1 del análisis de disparadores. La política es una red (10 → 14 → 4) y sus
parámetros ajustados son 214 pesos, no umbrales: una acción se enciende cuando
su salida sigmoide supera 0.5, y eso depende de la COMBINACIÓN de las 10
variables de observación. No existe "el valor de SGI que activa la desaladora";
existe el estado en cada decisión. Este script lo registra, para que los niveles
siguientes —umbral equivalente por variable y regla explícita aproximada— tengan
de dónde partir.

`decode_front.py` guardaba solo qué acción estaba encendida cada año. Aquí se
guarda además, para cada política, contexto y año de decisión:

    S   (n_pol, n_ctx, n_anios, 10)   las 10 variables tal como las ve la política
    PI  (n_pol, n_ctx, n_anios, K)    su salida sigmoide, ANTES del umbral de 0.5
    A   (n_pol, n_ctx, n_anios, K)    la acción resultante (0/1)

Las banderas built_* no vienen del rollout: son estado de la política, que vive
en el cierre de policy_fn. Por eso la salida se RECALCULA a partir del estado y
los pesos, replicando policy_fn, y se verifica año por año que reproduzca
exactamente las acciones del rollout. Si alguna no coincide, la reconstrucción
diverge de la política real y el resultado no sirve: se reporta el conteo.

PI y A difieren a propósito en un caso: la regla R1 del catálogo apaga la
desaladora costera cuando la completa está encendida, aunque su salida supere
0.5. Guardar ambas deja ver cuándo la política "quería" la costera.

Contextos:
    · los 27 estados del mundo de la optimización: donde la política aprendió
      sus disparadores (clima GCM × población × superficie)
    · los 3 contextos de verificación en WMS2Ma: para enlazar con esas corridas

Uso (en paralelo, un proceso por núcleo libre):
    python weap_dps/estados_disparo.py --pareto_dir results/iter02_frente --workers 5
    python weap_dps/estados_disparo.py --pareto_dir results/iter02_frente --limit 2   # prueba
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _desempaquetar(P, N, M, K):
    W1 = P[:N * M].reshape(N, M)
    b1 = P[N * M:N * M + M]
    W2 = P[N * M + M:N * M + M + M * K].reshape(M, K)
    b2 = P[N * M + M + M * K:]
    return W1, b1, W2, b2


def trabajador(args) -> int:
    from weap_dps.action_translator import (ACTION_NAMES_BINARY, init_built_state,
                                            policy_output_to_actions,
                                            update_built_state)
    from weap_dps.analyze_pareto import load as load_fronts
    from weap_dps.config_weap import (DECISION_YEARS, POLICY_STATE_FEATURES,
                                      SPIN_UP_YEARS, ZARR_TEMPLATE_PATH)
    from weap_dps.export_pareto_runs import CONTEXTOS, _no_dominadas, plantilla_de_clima
    from weap_dps.main_robust_weap import RobustPipeWEAP
    from weap_dps.scenario_builder import build_scenarios

    seeds, F, V, *_ , opt, _diag = load_fronts(args.pareto_dir)
    Fall = np.vstack([F[s] for s in seeds])
    Vall = np.vstack([V[s] for s in seeds])
    nd = _no_dominadas(Fall)
    if args.limit:
        nd = nd[:args.limit]
    mias = np.arange(len(nd))[args.worker::args.workers]

    pipe = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=1.0)
    surr = pipe.surrogate
    sow, et_sow = build_scenarios(surr, pipe.feature_names, pipe.X_template, n_climate=5)
    ctx = [(f"sow:{e}", X) for e, X in zip(et_sow, sow)]
    ctx += [(f"verif:{et}", plantilla_de_clima(surr, pipe.feature_names,
                                               pipe.X_template, rid))
            for rid, et in CONTEXTOS]

    N, M, K = len(POLICY_STATE_FEATURES), pipe.policy_M, pipe.policy_K
    T = DECISION_YEARS
    S = np.full((len(mias), len(ctx), T, N), np.nan, np.float32)
    PI = np.full((len(mias), len(ctx), T, K), np.nan, np.float32)
    A = np.zeros((len(mias), len(ctx), T, K), np.int8)
    discrepancias = 0
    decisiones = 0

    t0 = time.perf_counter()
    for j, k in enumerate(mias):
        P = Vall[nd[k]]
        W1, b1, W2, b2 = _desempaquetar(P, N, M, K)
        pol = pipe._build_policy_from_params(P)
        for c, (_, X) in enumerate(ctx):
            r = surr.rollout_with_policy(
                X_template=X, policy_fn=pol, n_years=T,
                action_col_idx=pipe.action_col_idx, spin_up_years=SPIN_UP_YEARS)
            H = np.asarray(r["actions_history"], dtype=float)[:, :K]
            built = init_built_state()
            for t, sd in enumerate(r["policy_states"]):
                # Réplica exacta de policy_fn (pipe_simulation_weap.py).
                s = np.array([built.get("act_" + f[6:], 0.0) if f.startswith("built_")
                              else sd.get(f, 0.0) for f in POLICY_STATE_FEATURES],
                             dtype=float)
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                pi = 1.0 / (1.0 + np.exp(-(np.tanh(s @ W1 + b1) @ W2 + b2)))
                act = policy_output_to_actions(pi)
                a = np.array([act[n] for n in ACTION_NAMES_BINARY])
                decisiones += 1
                if not np.array_equal(a > 0.5, H[t] > 0.5):
                    discrepancias += 1
                S[j, c, t], PI[j, c, t], A[j, c, t] = s, pi[:K], (H[t] > 0.5)
                built = update_built_state(built, act)
        if (j + 1) % 5 == 0 or j + 1 == len(mias):
            dt = time.perf_counter() - t0
            print(f"[w{args.worker}] {j+1}/{len(mias)} políticas | "
                  f"{dt/(j+1):.0f} s/política | faltan {(len(mias)-j-1)*dt/(j+1)/60:.0f} min | "
                  f"discrepancias {discrepancias}/{decisiones}", flush=True)

    parte = args.out.with_suffix(f".part{args.worker}.npz")
    parte.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        parte, pos=mias, S=S, PI=PI, A=A,
        F=Fall[nd[mias]], idx_front=nd[mias],
        ctx=np.array([e for e, _ in ctx], dtype=object),
        estado=np.array(POLICY_STATE_FEATURES, dtype=object),
        acciones=np.array(ACTION_NAMES_BINARY, dtype=object),
        objetivos=np.array(list(opt), dtype=object),
        anios=np.arange(2027, 2027 + T),
        discrepancias=discrepancias, decisiones=decisiones)
    print(f"[w{args.worker}] guardado {parte} | discrepancias {discrepancias}/{decisiones}",
          flush=True)
    return 0


def unir(out: Path, n: int) -> int:
    # Se copian a memoria y se cierran: np.load deja el .npz abierto de forma
    # perezosa, y en Windows eso impide borrar las partes después.
    partes = []
    for i in range(n):
        with np.load(out.with_suffix(f".part{i}.npz"), allow_pickle=True) as z:
            partes.append({k: z[k] for k in z.files})
    pos = np.concatenate([p["pos"] for p in partes])
    orden = np.argsort(pos)
    cat = lambda k: np.concatenate([p[k] for p in partes])[orden]
    disc = int(sum(int(p["discrepancias"]) for p in partes))
    dec = int(sum(int(p["decisiones"]) for p in partes))
    p0 = partes[0]
    np.savez_compressed(
        out, S=cat("S"), PI=cat("PI"), A=cat("A"), F=cat("F"),
        idx_front=cat("idx_front"), ctx=p0["ctx"], estado=p0["estado"],
        acciones=p0["acciones"], objetivos=p0["objetivos"], anios=p0["anios"],
        discrepancias=disc, decisiones=dec)
    for i in range(n):
        out.with_suffix(f".part{i}.npz").unlink()
    print(f"\nunido: {out}  | {len(pos)} políticas × {len(p0['ctx'])} contextos × "
          f"{len(p0['anios'])} años")
    print(f"consistencia con el rollout: {disc} discrepancias en {dec} decisiones"
          + ("  -> OK" if disc == 0 else "  -> REVISAR: la réplica diverge de policy_fn"))
    return 0 if disc == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pareto_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("results/estados_disparo_iter02.npz"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--limit", type=int, default=None,
                    help="solo las primeras N políticas (prueba)")
    args = ap.parse_args()

    if args.worker is not None:
        return trabajador(args)

    # Lanzador: un proceso por trabajador. Procesos y no hilos, con 1 hilo de
    # torch cada uno: es como corren las semillas del DPS y evita que se pisen.
    t0 = time.time()
    base = [sys.executable, __file__, "--pareto_dir", str(args.pareto_dir),
            "--out", str(args.out), "--workers", str(args.workers)]
    if args.limit:
        base += ["--limit", str(args.limit)]
    procs = [subprocess.Popen(base + ["--worker", str(i)], env=os.environ.copy())
             for i in range(args.workers)]
    codigos = [p.wait() for p in procs]
    if any(codigos):
        print(f"algún trabajador falló: {codigos}")
        return 1
    r = unir(args.out, args.workers)
    print(f"tiempo total: {(time.time()-t0)/60:.1f} min")
    return r


if __name__ == "__main__":
    raise SystemExit(main())
