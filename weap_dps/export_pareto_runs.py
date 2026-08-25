# -*- coding: utf-8 -*-
"""
export_pareto_runs.py — runs WEAP desde el frente de una iteración del DPS.

Paso 3 del ciclo de refinamiento (§5 de Metodología): las políticas del frente se
decodifican a cronogramas ejecutables y se simulan en WMMaS2 para medir la
discrepancia del emulador **sobre los objetivos de decisión**, no sobre variables
individuales.

DISEÑO: 25 políticas × 2 contextos climáticos = 50 runs
    · 25 políticas: 5 extremos (uno por objetivo) + 20 por k-means sobre el
      frente combinado de las 6 semillas, normalizado por rango.
    · 2 contextos que difieren SOLO en clima, con población y área en su valor
      base, de modo que el par sea comparable.

Por qué 25 × 2 y no 12 × 4: el ensamble de 900 runs ya cubre bien el clima
—8 series con 82 a 128 runs cada una, más 185 con sequía impuesta— y el emulador
reproduce la respuesta climática con r = 0,990 frente a WEAP (§1.3 de Resultados).
Gastar simulaciones de 70 min en la dimensión que el emulador ya domina rinde
poco. Lo que estos runs miden es el error sobre las políticas que el optimizador
propone, y para estimar esa distribución a lo largo del frente manda el número de
políticas distintas, no el de climas por política.

Los cronogramas se recalculan PARA CADA CONTEXTO. La política es de lazo cerrado:
usar el mismo cronograma en los dos climas simularía una política de lazo abierto
disfrazada, y perdería justamente lo que se quiere verificar.

Uso:
    python weap_dps/export_pareto_iter2.py \
        --pareto_dir runs_weap/robust_iter1_fix2050 \
        --iteration 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weap_dps.analyze_pareto import load as load_fronts
from weap_dps.config_weap import (DATA_DIR, DECISION_YEARS, TRAIN_ZARR_PATH,
                                  ZARR_TEMPLATE_PATH)
from weap_dps.pareto_to_runids import (evaluate_policy_to_schedule,
                                       schedule_to_weap_csv)
from weap_dps.scenario_builder import (PP_ACUM_SOURCE, PP_ACUM_WINDOWS,
                                       SUBCUENCAS, _normalize_col, _rolling_sum)

# Contextos: (run_id del zarr, etiqueta). Precipitación MEDIA ANUAL medida sobre
# 2014-2060 en las 6 subcuencas, no acumulada a 104 semanas.
#
# Ambos contextos son SEQUÍAS, de severidad creciente: uno en sintonía con la
# megasequía observada y otro más seco todavía, para tensionar el sistema.
#
#   882  AWI-CM-1-1-MR ssp585, sev 0.70 · 2035-2045 — 176.3 mm/año en el
#        conjunto y 112.5 durante la década (caída del 42%). Es la caída más
#        moderada de las cuatro sequías del ensamble y por tanto la más parecida
#        a una megasequía prolongada. Cae dentro del horizonte con margen: ocho
#        años de operación previa y quince de recuperación, de modo que la
#        respuesta de lazo cerrado se verifica antes, durante y después.
#   693  AWI-CM-1-1-MR ssp585, sev 0.85 · 2040-2050 — 170.8 mm/año, la más baja
#        del ensamble DPS, con 56.3 durante la década (caída del 72%).
#
# Referencia de los cinco GCM sin sequía impuesta: 130.5 · 158.4 · 184.8 · 201.0
# · 234.1 mm/año. Ambos contextos quedan por debajo de la mediana (184.8).
#
# Se descartó el run 526 (sev 0.72 desde 2025), que parecía mejor por arrancar
# antes: su clima base es de los más húmedos y termina en 206.8 mm/año, MÁS
# lluvia que el GCM mediano, de modo que el "contexto seco" habría sido más
# húmedo que el normal.
#
# No se incluye un contexto de clima normal: el ensamble de 900 runs ya cubre
# ampliamente las condiciones GCM (8 series, 82 a 128 runs cada una) y es la
# región donde el emulador mejor reproduce la respuesta climática (r = 0.990).
# La escasez está en la verificación bajo estrés hídrico sostenido.
CONTEXTOS = [(882, "megasequia"), (693, "sequia_extrema")]


def _no_dominadas(F: np.ndarray) -> np.ndarray:
    keep = []
    for i in range(len(F)):
        if not any((F[j] <= F[i]).all() and (F[j] < F[i]).any()
                   for j in range(len(F)) if j != i):
            keep.append(i)
    return np.array(keep)


def seleccionar(F: np.ndarray, V: np.ndarray, n_total: int) -> list[tuple[int, str]]:
    """5 extremos (mínimo de cada objetivo) + resto por k-means."""
    nd = _no_dominadas(F)
    Fn, idx_map = F[nd], nd
    sel: list[tuple[int, str]] = []
    usados: set[int] = set()
    for k in range(Fn.shape[1]):
        for cand in np.argsort(Fn[:, k]):
            if int(cand) not in usados:
                usados.add(int(cand))
                sel.append((int(idx_map[cand]), f"extremo_J{k+1}"))
                break
    resto = [i for i in range(len(Fn)) if i not in usados]
    n_bal = n_total - len(sel)
    if resto and n_bal > 0:
        rango = np.maximum(Fn.max(0) - Fn.min(0), 1e-9)
        Z = (Fn[resto] - Fn.min(0)) / rango
        km = KMeans(n_clusters=min(n_bal, len(resto)), n_init=10,
                    random_state=42).fit(Z)
        for c in range(km.n_clusters):
            m = np.where(km.labels_ == c)[0]
            if not len(m):
                continue
            d = np.linalg.norm(Z[m] - km.cluster_centers_[c], axis=1)
            sel.append((int(idx_map[resto[m[int(np.argmin(d))]]]), f"balanceada_{c+1}"))
    return sel[:n_total]


def plantilla_de_clima(surrogate, feature_names, base: np.ndarray,
                       run_id: int) -> np.ndarray:
    """Inyecta precip/temp del run indicado y RECALCULA las acumuladas.

    Omitir el recálculo de las derivadas deja la dimensión climática inerte
    aunque el forzante base sí cambie (§4.3 de Metodología): es un modo de falla
    silencioso que ninguna métrica de ajuste detecta.
    """
    fi = {n: i for i, n in enumerate(feature_names)}
    Z = zarr.open_group(str(TRAIN_ZARR_PATH), mode="r")
    zfeat = list(Z.attrs["feature_names"]); zfi = {n: i for i, n in enumerate(zfeat)}
    zr = np.asarray(Z["run_ids"][:]).astype(int)
    if run_id not in set(zr.tolist()):
        raise SystemExit(f"El run climático {run_id} no está en {TRAIN_ZARR_PATH.name}")
    raw = Z["X"][int(np.where(zr == run_id)[0][0])]
    T = base.shape[0]
    X = base.copy()
    for col in [f"Precipitation__{s}" for s in SUBCUENCAS] + \
               [f"Temperature__{s}" for s in SUBCUENCAS]:
        if col in fi and col in zfi:
            X[:, fi[col]] = _normalize_col(surrogate, raw[:T, zfi[col]], fi[col])
    if PP_ACUM_SOURCE in zfi:
        pp = raw[:T, zfi[PP_ACUM_SOURCE]].astype(float)
        for w in PP_ACUM_WINDOWS:
            col = f"PP_acum_{w}weeks"
            if col in fi:
                X[:, fi[col]] = _normalize_col(surrogate, _rolling_sum(pp, w), fi[col])
    return X


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pareto_dir", type=Path, required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--n_pol", type=int, default=25)
    ap.add_argument("--start_id", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    start_id = args.start_id if args.start_id is not None \
        else 2000 + (args.iteration - 1) * 100
    out = args.out or (DATA_DIR / "exports" / f"iter_{args.iteration:02d}")
    pol_dir = out / "policies"
    pol_dir.mkdir(parents=True, exist_ok=True)

    seeds, F, V, A, el, names, opt, diag = load_fronts(args.pareto_dir)
    Fall = np.vstack([F[s] for s in seeds])
    Vall = np.vstack([V[s] for s in seeds])
    print(f"frente: {len(Fall)} soluciones de {len(seeds)} semillas")

    sel = seleccionar(Fall, Vall, args.n_pol)
    print(f"seleccionadas {len(sel)} políticas "
          f"({sum(1 for _, r in sel if r.startswith('extremo'))} extremos)")

    from weap_dps.main_robust_weap import RobustPipeWEAP
    pipe = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=1.0)
    surr = pipe.surrogate

    plantillas = {et: plantilla_de_clima(surr, pipe.feature_names,
                                         pipe.X_template, rid)
                  for rid, et in CONTEXTOS}
    master, rid_actual = [], start_id
    m900 = pd.read_csv(Path(__file__).resolve().parents[2] /
                       "WEAP_2_ZARR" / "data" / "RunIDs_Q_full.csv").set_index("ID")

    class _Sol:                       # evaluate_policy_to_schedule espera .variables
        def __init__(self, v): self.variables = v

    for k, (i, rol) in enumerate(sel):
        for run_clima, etiqueta in CONTEXTOS:
            # Cronograma RECALCULADO para este contexto (política de lazo cerrado).
            hist = evaluate_policy_to_schedule(_Sol(Vall[i]), surr, pipe,
                                               climate_template=plantillas[etiqueta])
            nombre = f"policy_iter{args.iteration:02d}_{rid_actual:04d}_{rol}_{etiqueta}.csv"
            schedule_to_weap_csv(hist, start_year=2027,
                                 output_path=pol_dir / nombre,
                                 include_q_values=True, hydro_end_year=2060)
            base = m900.loc[run_clima]
            master.append({
                "ID": rid_actual, "GCM": base.GCM, "SSP": base.SSP,
                "Demanda_Agro": base.Demanda_Agro,
                "Demanda_Poblacion": base.Demanda_Poblacion,
                "drought_severity": base.drought_severity,
                "drought_duration": base.drought_duration,
                "drought_start_year": base.drought_start_year,
                "drought_severity_mode": base.drought_severity_mode,
                "block": f"dps_proposal_iter{args.iteration}",
                "policy_file": nombre,
                "pareto_idx": int(i), "role": rol, "contexto": etiqueta,
                "clima_run": run_clima,
                "description": (f"DPS iter{args.iteration} {rol} | contexto {etiqueta} "
                                f"(run {run_clima}) | {base.GCM} {base.SSP}"),
            })
            rid_actual += 1
        if (k + 1) % 5 == 0:
            print(f"  {k+1}/{len(sel)} políticas")

    df = pd.DataFrame(master)
    csv = out / f"RunIDs_Q_pareto_iter{args.iteration:02d}.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    (out / "seleccion.json").write_text(json.dumps(
        {"n_politicas": len(sel), "contextos": CONTEXTOS,
         "objetivos": list(names), "start_id": start_id,
         "roles": [r for _, r in sel]}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(df)} runs escritos")
    print(f"  master   : {csv}")
    print(f"  políticas: {pol_dir}")
    print(f"  IDs      : {start_id}-{rid_actual-1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
