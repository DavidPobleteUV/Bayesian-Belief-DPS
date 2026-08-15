# -*- coding: utf-8 -*-
"""
robustness_test.py — Verificación de robustez de segunda etapa.

El frente de Pareto se obtiene optimizando sobre un ensamble ACOTADO (27 estados
del mundo). Eso basta para *encontrar* políticas eficientes, pero no para
afirmar que son robustas: con 27 SOWs la desviación estándar que entra en el
objetivo robusto tiene error de muestreo grande, y cada clima aparece 3 veces.

Este script hace la segunda etapa del procedimiento estándar en RDM: toma el
frente YA OBTENIDO y lo re-evalúa sobre un ensamble MAYOR, sin reoptimizar. Es
órdenes de magnitud más barato que optimizar sobre el ensamble grande, y separa
dos preguntas que conviene no mezclar:

    ¿la búsqueda encontró buenas políticas?   -> convergencia (analyze_pareto.py)
    ¿esas políticas aguantan?                 -> robustez (este script)

Métricas (Herman et al. 2015, "How should robustness be defined..."):

  - CRITERIO DE DOMINIO (satisficing): fracción de SOWs donde la política
    cumple todos los umbrales. Es el criterio que entiende una sanitaria.
  - ARREPENTIMIENTO MÁXIMO (regret): peor diferencia, sobre todos los SOWs,
    entre lo que logra esta política y lo mejor que se podía lograr en ESE SOW.
    Penaliza a las políticas que fallan justo cuando importa.
  - PEOR CASO: el valor más desfavorable de cada objetivo sobre el ensamble.

Uso:
    python weap_dps/robustness_test.py runs_weap/robust_iter2_sow27 \
        --n_sow 81 --n_pol 40 --out results/robustez_iter2

IMPORTANTE: necesita el zarr COMPLETO (el subset solo trae los 9 climas de la
optimización). Se apunta con $env:DPS_TRAIN_ZARR o con --zarr.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Umbrales del criterio de dominio. Son juicios de política, no del modelo:
# se declaran acá para que sean explícitos y discutibles.
UMBRALES = {
    # el déficit del peor año no debe superar este múltiplo de la demanda anual
    "J52_worst_year_frac": 1.15,
    # semanas en falla por pueblo, promedio, sobre 1716 del horizonte
    "J51_mean_town_fail": 900.0,
}


def cargar_frente(d: Path):
    """Políticas y objetivos del frente combinado de todas las semillas."""
    V, F, names, opt = [], [], None, None
    for f in sorted(d.glob("pareto_seed*.dat")):
        obj = pickle.load(open(f, "rb"))
        V += [np.array(v, float) for v, _ in obj["result"]]
        F += [np.array(o, float) for _, o in obj["result"]]
        names = obj.get("objective_names", names)
        opt = obj.get("objectives_optimized", opt)
    if not V:
        raise SystemExit(f"No hay pareto_seed*.dat en {d}")
    return np.array(V), np.array(F), names, opt


def ensamble_amplio(n_sow: int, climas_extra: list[int]):
    """Diseño balanceado sobre un conjunto de climas MAYOR que el de la optimización.

    Incluye a propósito escenarios de estrés que se excluyeron de la búsqueda
    (sequías largas y severas): la pregunta acá no es qué política es óptima en
    promedio, sino cuáles aguantan lo que no se optimizó.
    """
    from weap_dps.config_weap import DPS_CLIMATE_RUNS, POP_LEVELS, AREA_LEVELS
    climas = list(dict.fromkeys(list(DPS_CLIMATE_RUNS) + list(climas_extra)))
    combos = list(itertools.product(POP_LEVELS.items(), AREA_LEVELS.items()))
    paso = 5 if len(combos) % 5 else 7
    sow = []
    for i in range(n_sow):
        c = climas[i % len(climas)]
        (pn, pv), (an, av) = combos[(i * paso) % len(combos)]
        sow.append((c, pn, pv, an, av))
    return sow, climas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path, help="carpeta con pareto_seed*.dat")
    ap.add_argument("--n_sow", type=int, default=81)
    ap.add_argument("--n_pol", type=int, default=40,
                    help="políticas a evaluar (muestra del frente)")
    ap.add_argument("--climas_extra", type=int, nargs="*",
                    default=[555, 461, 463, 560, 421, 495, 693, 779, 882],
                    help="runs climáticos adicionales, incluidos los de estrés "
                         "que se dejaron FUERA de la optimización")
    ap.add_argument("--zarr", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/robustez"))
    a = ap.parse_args()

    if a.zarr:
        os.environ["DPS_TRAIN_ZARR"] = a.zarr
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from weap_dps.config_weap import (ZARR_TEMPLATE_PATH, TRAIN_ZARR_PATH,
                                      DECISION_YEARS, SPIN_UP_YEARS, WARMUP_WEEKS,
                                      WEEKS_PER_YEAR, ACTION_NAMES_BINARY,
                                      ACTION_NAMES_QUANTITY, OBJECTIVE_NAMES,
                                      j4_calibration_factor)
    from weap_dps.pipe_simulation_weap import PipeWEAP
    from weap_dps.scenario_builder import build_scenarios
    from weap_dps.cost_calculator import compute_objectives

    W0 = WARMUP_WEEKS + SPIN_UP_YEARS * WEEKS_PER_YEAR
    print(f"zarr: {TRAIN_ZARR_PATH}")

    V, F, names, opt = cargar_frente(a.dir)
    print(f"frente: {len(V)} políticas, {V.shape[1]} parámetros")

    pipe = PipeWEAP(ZARR_TEMPLATE_PATH)
    sow, climas = ensamble_amplio(a.n_sow, a.climas_extra)
    faltan = [c for c in climas if c not in
              set(np.array(__import__("zarr").open_group(str(TRAIN_ZARR_PATH),
                                                         mode="r")["run_ids"][:]).astype(int).tolist())]
    if faltan:
        raise SystemExit(
            f"Faltan los climas {faltan} en el zarr. Este análisis necesita el "
            f"zarr COMPLETO: pásalo con --zarr o $env:DPS_TRAIN_ZARR.")

    # se reutiliza build_scenarios inyectando el diseño ampliado
    import weap_dps.scenario_builder as sb
    orig = sb.build_sow_design
    sb.build_sow_design = lambda climate_runs, n: sow
    scen, labels = build_scenarios(pipe.surrogate, pipe.feature_names,
                                   pipe.X_template, n_climate=len(climas))
    sb.build_sow_design = orig
    print(f"ensamble de verificación: {len(scen)} SOWs sobre {len(climas)} climas "
          f"(la optimización usó 27 sobre 9)")

    # muestra del frente, repartida a lo largo del eje de costo
    i_cost = opt.index("J4_supply_cost") if opt and "J4_supply_cost" in opt else 2
    order = np.argsort(F[:, i_cost])
    sel = order[np.linspace(0, len(order) - 1, min(a.n_pol, len(order))).astype(int)]
    print(f"políticas evaluadas: {len(sel)}   "
          f"rollouts: {len(sel) * len(scen)}  "
          f"(~{len(sel) * len(scen) * 1.9 / 3600:.1f} h)\n")

    res = np.zeros((len(sel), len(scen), len(OBJECTIVE_NAMES)))
    for i, k in enumerate(sel):
        fn = pipe._build_policy_from_params(V[k])
        for j, X in enumerate(scen):
            r = pipe.surrogate.rollout_with_policy(
                X_template=X, policy_fn=fn, n_years=DECISION_YEARS,
                action_col_idx=pipe.action_col_idx, spin_up_years=SPIN_UP_YEARS)
            sf = pipe.correct_balance(
                pipe.surrogate.denormalize_y(r["surface"], kind="surface"), X)
            o = compute_objectives(
                gw_denorm=pipe.surrogate.denormalize_y(r["gw"], kind="gw"),
                surf_denorm=sf, target_names_gw=pipe.target_names_gw,
                target_names_surf=pipe.target_names_surf,
                actions_history=r["actions_history"],
                action_names_order=ACTION_NAMES_BINARY + ACTION_NAMES_QUANTITY,
                decision_start_week=W0, ap_demand_m3s=pipe._ap_demand(X),
                ap_town_order=pipe.ap_town_order)
            v = np.array(list(o.values()), float)
            ah = np.asarray(r["actions_history"])
            n_act = int((ah[:, :len(ACTION_NAMES_BINARY)] > 0.5).any(axis=0).sum())
            v[OBJECTIVE_NAMES.index("J4_supply_cost")] *= j4_calibration_factor(n_act)
            res[i, j] = v
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(sel)} políticas")

    # ── métricas de robustez ──
    idx = {n: i for i, n in enumerate(OBJECTIVE_NAMES)}
    cumple = np.ones((len(sel), len(scen)), bool)
    for nm, thr in UMBRALES.items():
        cumple &= res[:, :, idx[nm]] <= thr
    dominio = cumple.mean(axis=1)

    # arrepentimiento sobre los objetivos que se optimizaron (todos a minimizar)
    oi = [idx[n] for n in (opt or OBJECTIVE_NAMES)]
    reg = np.zeros((len(sel), len(oi)))
    for c, k in enumerate(oi):
        Y = res[:, :, k]
        mejor = Y.min(axis=0)                       # mejor politica en cada SOW
        rango = np.maximum(Y.max(axis=0) - mejor, 1e-12)
        reg[:, c] = ((Y - mejor) / rango).max(axis=1)   # peor arrepentimiento
    regret_max = reg.max(axis=1)

    print(f"\n=== ROBUSTEZ SOBRE {len(scen)} SOWs ===")
    print(f"umbrales del criterio de dominio: {UMBRALES}\n")
    print(f"{'#':>3} {'dominio':>9} {'regret':>8} " +
          "  ".join(f"{n.split('_')[0]:>10}" for n in (opt or [])))
    orden = np.argsort(-dominio)
    for c in orden[:15]:
        med = [np.median(res[c, :, idx[n]]) for n in (opt or [])]
        print(f"{c:3d} {100*dominio[c]:8.1f}% {regret_max[c]:8.3f} " +
              "  ".join(f"{v:10.3g}" for v in med))

    print(f"\ndominio: mediana {100*np.median(dominio):.1f}%   "
          f"mejor {100*dominio.max():.1f}%   peor {100*dominio.min():.1f}%")
    print(f"políticas que cumplen en >90% de los SOWs: {(dominio > 0.9).sum()}/{len(sel)}")
    print(f"políticas que cumplen en <50%            : {(dominio < 0.5).sum()}/{len(sel)}")

    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "robustez.npz", res=res, sel=sel, dominio=dominio,
             regret=regret_max, labels=np.array(labels, dtype=object))
    (a.out / "resumen.json").write_text(json.dumps({
        "n_sow": len(scen), "n_climas": len(climas), "n_politicas": len(sel),
        "umbrales": UMBRALES,
        "dominio_mediana": float(np.median(dominio)),
        "dominio_max": float(dominio.max()),
        "n_robustas_90": int((dominio > 0.9).sum()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nguardado en {a.out}")


if __name__ == "__main__":
    main()
