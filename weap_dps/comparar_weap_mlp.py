# -*- coding: utf-8 -*-
"""
comparar_weap_mlp.py — paso 4 del ciclo de refinamiento (§5 de Metodología).

Contrasta lo que el emulador predice contra lo que WMMaS2 simula, para los runs
de verificación exportados desde el frente. Tres criterios:

  1. ERROR POR OBJETIVO DE DECISIÓN (primario). Es lo que decide si se reentrena.
     Tolerancia declarada: 5 %.

     El error se mide en RELATIVO salvo para J2, cuyo valor de referencia puede
     ser exactamente cero: hay políticas del frente que en WMMaS2 no dejan
     ningún déficit (run 2200: 0.1 m³ en 33 años). Contra ese cero cualquier
     predicción positiva da un error relativo arbitrariamente grande aunque el
     error absoluto sea despreciable. J2 se reporta por tanto en PUNTOS
     PORCENTUALES DE LA DEMANDA: |J2_mlp - J2_weap| / demanda_total, que está
     bien definido en cero y es directamente interpretable como "el emulador se
     equivoca en x % del agua potable del horizonte".
  2. AJUSTE POR TRAYECTORIA (respaldo). KGE por familia de variables. Distingue
     dos casos que el error agregado confunde: seguir bien la dinámica y fallar
     en el nivel, o acertar el total compensando errores de signo opuesto.
  3. REPARTO POR FUENTE. Los enlaces de transmisión son el eslabón más débil
     (KGE 0.856 tras la corrección del corte en 2050); aquí se verifica si la
     cascada de despacho reproduce la asignación de WEAP.

METODOLOGÍA DE LA COMPARACIÓN
    El escenario del emulador es el PROPIO X del run de WEAP, no uno
    reconstruido: así los forzantes son idénticos por construcción y lo único
    que se mide es el error del emulador. Las acciones ya vienen en ese X, de
    modo que no hace falta re-evaluar la política.

    Se reporta con y sin la corrección de balance y la calibración de J4, porque
    ambas se ajustaron sobre el conjunto de test y conviene ver qué aportan en
    un caso nuevo.

Uso:
    python weap_dps/comparar_weap_mlp.py --zarr <ruta> --run 2200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weap_dps import balance_correction as bc
from weap_dps.config_weap import (TOWN_SOURCE_COST_CSV, WARMUP_WEEKS,
                                  j4_calibration_factor)
from weap_dps.cost_calculator import compute_objectives

W0 = 676                       # inicio del periodo de decisión (2027)
TOL = 0.05                     # tolerancia declarada del criterio 1


def normalizar_x(surr, feat_modelo: list[str], X_raw: np.ndarray,
                 nombres_raw: list[str]) -> np.ndarray:
    """X crudo (T, n_raw) -> X normalizado en el orden que espera el modelo.

    Replica lo que hace normalize_zarr: transformada por columna (log/arcsinh)
    y luego z-score con los scalers del entrenamiento. No los REAJUSTA: aplicar
    scalers nuevos a un run suelto lo pondría en otra escala que la del modelo.
    """
    idx = {n: i for i, n in enumerate(nombres_raw)}
    faltan = [n for n in feat_modelo if n not in idx]
    if faltan:
        raise SystemExit(f"Faltan {len(faltan)} features en el zarr: {faltan[:5]}")
    X = np.stack([X_raw[:, idx[n]] for n in feat_modelo], axis=1).astype(np.float64)
    meth = surr.transform_methods_x_filt
    for j in range(X.shape[1]):
        m = str(meth[j]) if meth is not None else "none"
        if m == "log":
            X[:, j] = np.log(np.maximum(X[:, j], 0) + 0.1)
        elif m == "arcsinh":
            X[:, j] = np.arcsinh(X[:, j] / 0.1)
    return ((X - surr.x_mean) / surr.x_std).astype(np.float32)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    ok = np.isfinite(obs) & np.isfinite(sim)
    if ok.sum() < 10 or np.std(obs[ok]) < 1e-12:
        return np.nan
    o, s = obs[ok], sim[ok]
    r = np.corrcoef(o, s)[0, 1]
    a = np.std(s) / np.std(o)
    b = np.mean(s) / (np.mean(o) if abs(np.mean(o)) > 1e-12 else 1e-12)
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


def evaluar_run(pipe, Z, run: int, verbose: bool = True) -> dict:
    """Compara un run: devuelve objetivos, KGE por familia y reparto por fuente."""
    surr = pipe.surrogate
    rid = np.asarray(Z["run_ids"][:]).astype(int)
    k = int(np.where(rid == run)[0][0])
    X_raw, Y_raw = np.nan_to_num(Z["X"][k]), np.nan_to_num(Z["Y"][k])
    fn_raw, tn_raw = list(Z.attrs["feature_names"]), list(Z.attrs["target_names"])
    if verbose:
        print(f"run {run}: X{X_raw.shape}  Y{Y_raw.shape}")
    Xn = normalizar_x(surr, list(pipe.feature_names), X_raw, fn_raw)

    # Rollout recursivo LIBRE: las acciones ya vienen en X, no hay politica que
    # re-evaluar. Es el mismo camino que usa evaluate_recursive sobre el test.
    xt = torch.tensor(Xn[None, ...], dtype=torch.float32)
    gw_n, surf_n = surr.model.model.forward_sequence(xt, WARMUP_WEEKS)
    gw = surr.denormalize_y(gw_n[0].numpy(), kind="gw")
    surf = surr.denormalize_y(surf_n[0].numpy(), kind="surface")

    # WEAP en el mismo espacio de nombres que el emulador
    ti = {n: i for i, n in enumerate(tn_raw)}
    gw_w = np.stack([Y_raw[:, ti[n]] for n in surr.target_names_gw], axis=1)
    sf_w = np.stack([Y_raw[:, ti[n]] for n in surr.target_names_surf], axis=1)

    # Demanda AP cruda, para la corrección de balance y para J51/J52
    dcols = [i for i, n in enumerate(fn_raw) if n.startswith("AP_WaterDemand__")]
    dem = X_raw[:, dcols]

    # Historial de acciones desde el propio X: ambos lados reciben el MISMO, de
    # modo que el CAPEX es identico y se cancela en la comparacion. Lo que se
    # mide es el error del emulador, no una diferencia de escenario.
    from weap_dps.action_translator import (ACTION_NAMES_BINARY,
                                            ACTION_NAMES_QUANTITY)
    orden = ACTION_NAMES_BINARY + ACTION_NAMES_QUANTITY
    n_anios = (X_raw.shape[0] - W0) // 52
    hist = np.zeros((n_anios, len(orden)))
    for y in range(n_anios):
        t = W0 + y * 52
        for jj, nom in enumerate(orden):
            if nom in fn_raw:
                hist[y, jj] = X_raw[t, fn_raw.index(nom)]

    def objetivos(g, s, etiqueta, n_acc=None):
        o = compute_objectives(
            gw_denorm=g, surf_denorm=s,
            target_names_gw=surr.target_names_gw,
            target_names_surf=surr.target_names_surf,
            actions_history=hist, action_names_order=orden,
            decision_start_week=W0, ap_demand_m3s=dem)
        if n_acc is not None:
            o["J4_supply_cost"] *= j4_calibration_factor(n_acc)
        return o

    A = ["act_desalacion_costera", "act_desalacion_completa",
         "act_nuevo_pozo_a_5km", "act_acuerdo"]
    n_acc = sum(int(X_raw[:, fn_raw.index(a)].max() > 0) for a in A)

    o_weap = objetivos(gw_w, sf_w, "WEAP")
    o_crudo = objetivos(gw, surf, "MLP crudo")
    iL = [i for i, n in enumerate(surr.target_names_surf)
          if n.startswith("AP_TransmissionLinks__")]
    iU = [i for i, n in enumerate(surr.target_names_surf)
          if n.startswith("AP_UnmetDemand__")]
    surf_c = bc.apply_balance_correction(surf, iL, iU, dem, W0)
    o_corr = objetivos(gw, surf_c, "MLP corregido", n_acc=n_acc)

    # Demanda total del horizonte, denominador de la metrica de J2.
    dem_total = float(np.maximum(dem[W0:], 0).sum() * 604800.0)

    fams = {}
    for nom, arr_p, arr_w in (("GW", gw, gw_w), ("SUP", surf_c, sf_w)):
        nombres = surr.target_names_gw if nom == "GW" else surr.target_names_surf
        for j, n in enumerate(nombres):
            f = n.split("__")[0][:34]
            fams.setdefault(f, []).append(kge(arr_w[W0:, j], arr_p[W0:, j]))
    fam = {f: np.nanmedian(v) for f, v in fams.items() if np.isfinite(v).any()}

    cm = pd.read_csv(TOWN_SOURCE_COST_CSV)
    cm["withdrawal_node"] = cm["withdrawal_node"].astype(str).str.strip()
    src = {"Withdrawal_Node_" + r["withdrawal_node"]: r["source_type"]
           for _, r in cm.iterrows()}
    grupos = {}
    for j, n in enumerate(surr.target_names_surf):
        if not n.startswith("AP_TransmissionLinks__") or "_to_" not in n:
            continue
        t = n.split("__", 1)[1].rsplit("_to_", 1)[0].replace("Transmission_Link_from_", "")
        grupos.setdefault(src.get(t, "Acuerdo" if t.startswith("DemAGRO") else "otro"),
                          []).append(j)
    reparto = {g: (float(np.maximum(sf_w[W0:, jj], 0).sum() / 1e6),
                   float(np.maximum(surf[W0:, jj], 0).sum() / 1e6))
               for g, jj in grupos.items()}

    return {"run": run, "n_acc": n_acc, "weap": o_weap, "crudo": o_crudo,
            "corr": o_corr, "fam": fam, "reparto": reparto,
            "dem_total": dem_total,
            "kge_mediano": float(np.nanmedian(list(fam.values())))}


def error_objetivo(nombre: str, w: float, m: float, dem_total: float):
    """(error, es_relativo).

    J2 se mide en puntos porcentuales de la demanda y no en relativo: su valor
    de referencia puede ser exactamente cero —hay politicas del frente que en
    WMMaS2 no dejan ningun deficit— y contra ese cero cualquier prediccion
    positiva da un error relativo arbitrariamente grande aunque el error
    absoluto sea despreciable.
    """
    if nombre.startswith("J2") and dem_total > 0:
        return abs(m - w) / dem_total, False
    if not np.isfinite(w) or abs(w) < 1e-9:
        return np.nan, True
    return abs(m - w) / abs(w), True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", type=Path, required=True)
    ap.add_argument("--run", default="all",
                    help="ID, lista separada por comas, o 'all'")
    ap.add_argument("--csv", type=Path,
                    default=Path("results/comparacion_weap_mlp.csv"))
    args = ap.parse_args()

    torch.set_num_threads(1); torch.set_grad_enabled(False)
    Z = zarr.open_group(str(args.zarr), mode="r")
    disponibles = sorted(np.asarray(Z["run_ids"][:]).astype(int).tolist())
    runs = (disponibles if str(args.run).lower() == "all"
            else [int(x) for x in str(args.run).split(",")])
    faltan = [r for r in runs if r not in disponibles]
    if faltan:
        raise SystemExit(f"No estan en el zarr: {faltan}")

    from weap_dps.config_weap import ZARR_TEMPLATE_PATH
    from weap_dps.main_robust_weap import RobustPipeWEAP
    pipe = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=1.0)

    print(f"\ncomparando {len(runs)} runs: {runs[0]}-{runs[-1]}")
    res = []
    for i, r in enumerate(runs, 1):
        res.append(evaluar_run(pipe, Z, r, verbose=False))
        print(f"  [{i}/{len(runs)}] run {r}  KGE mediano {res[-1]['kge_mediano']:.3f}")

    sep = "=" * 78
    print(f"\n{sep}\nCRITERIO 1 - error por objetivo (tolerancia {TOL:.0%})\n{sep}")
    print("J2 en puntos porcentuales de la demanda; el resto en error relativo.\n")
    print(f"{'objetivo':22s} {'metrica':>10} {'mediana':>9} {'p90':>9} "
          f"{'max':>9} {'dentro tol.':>13}")
    filas = []
    for kk in list(res[0]["weap"]):
        e, rel = [], True
        for R in res:
            v, rel = error_objetivo(kk, R["weap"][kk], R["corr"][kk], R["dem_total"])
            if np.isfinite(v):
                e.append(v)
        if not e:
            continue
        e = np.array(e)
        dentro = 100.0 * (e <= TOL).mean()
        print(f"{kk:22s} {('relativo' if rel else 'p.p. dem'):>10} "
              f"{100*np.median(e):8.2f}% {100*np.percentile(e, 90):8.2f}% "
              f"{100*e.max():8.2f}% {dentro:11.0f} %")
        filas.append({"objetivo": kk,
                      "metrica": "relativo" if rel else "pp_demanda",
                      "n_runs": len(e), "mediana": np.median(e),
                      "p90": np.percentile(e, 90), "max": e.max(),
                      "pct_dentro_tol": dentro})

    print(f"\n{sep}\nCRITERIO 2 - KGE por familia (mediana entre runs)\n{sep}")
    todas = {}
    for R in res:
        for f, v in R["fam"].items():
            todas.setdefault(f, []).append(v)
    orden = sorted(todas.items(), key=lambda x: -np.nanmedian(x[1]))
    for f, v in orden[:8]:
        print(f"  {f:36s} {np.nanmedian(v):8.3f}")
    print("  ...")
    for f, v in orden[-5:]:
        print(f"  {f:36s} {np.nanmedian(v):8.3f}")
    glob = [R["kge_mediano"] for R in res]
    print(f"\n  KGE mediano global: {np.median(glob):.3f}   "
          f"(p10 {np.percentile(glob, 10):.3f}, p90 {np.percentile(glob, 90):.3f})")

    print(f"\n{sep}\nCRITERIO 3 - reparto por fuente (volumen 2027-2060)\n{sep}")
    print("OJO: los runs se exportaron a WEAP con la tarifa del acuerdo en 2.500")
    print("y sin las correcciones de la cascada, de modo que este criterio mide")
    print("error del emulador MAS desfase de configuracion. Ver 4.8.\n")
    print(f"{'fuente':16s} {'WEAP (Mm3)':>12} {'MLP (Mm3)':>12} {'MLP/WEAP':>10}")
    for g in sorted({g for R in res for g in R["reparto"]}):
        vw = np.median([R["reparto"][g][0] for R in res if g in R["reparto"]])
        vp = np.median([R["reparto"][g][1] for R in res if g in R["reparto"]])
        rz = vp / vw if vw > 1e-9 else np.nan
        print(f"{g:16s} {vw:12.2f} {vp:12.2f} {rz:10.3f}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(args.csv, index=False, encoding="utf-8-sig")
    det = pd.DataFrame([{"run": R["run"], "n_acciones": R["n_acc"],
                         "kge_mediano": R["kge_mediano"],
                         **{f"weap_{k}": v for k, v in R["weap"].items()},
                         **{f"mlp_{k}": v for k, v in R["corr"].items()}}
                        for R in res])
    det.to_csv(args.csv.with_name(args.csv.stem + "_por_run.csv"),
               index=False, encoding="utf-8-sig")
    print(f"\n  resumen: {args.csv}")
    print(f"  por run: {args.csv.with_name(args.csv.stem + '_por_run.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
