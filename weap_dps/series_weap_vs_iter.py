# -*- coding: utf-8 -*-
"""
series_weap_vs_iter.py — series de tiempo: WMS2Ma contra las dos iteraciones.

Complementa a `figuras_weap_vs_iter.py`. Aquella compara el VALOR AGREGADO de
cada objetivo; esta compara la TRAYECTORIA de la variable que lo define. La
distincion importa: un emulador puede acertar el total del horizonte compensando
errores de signo opuesto a lo largo del tiempo, y eso solo se ve en la serie.

Una figura por variable, con tres corridas del frente como paneles. Se agregan a
resolucion ANUAL porque es la del descuento de los objetivos y la de la decision
de la politica; la semanal satura el grafico sin agregar informacion.

Uso:
    python weap_dps/series_weap_vs_iter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weap_dps.comparar_weap_mlp import normalizar_x

ZARR = (r"C:\Users\David\Documents\GitHub_DPL\WEAP_2_ZARR\results"
        r"\training_data\iter01_pareto\weap_weekly.zarr")
ART = {"iter01": "data_weap", "iter02": "data_weap_iter02"}
RUNS = [2200, 2210, 2225]           # una politica sin deficit, dos con
W0, WPY = 676, 52
OUT = Path("results/figuras")
COL = {"WMS2Ma": "#1a1a1a", "iter01": "#c0392b", "iter02": "#2874a6"}

# (id, titulo, patron de columnas, unidad, factor, agregacion anual)
VARS = [
    ("S1_almacenamiento", "Almacenamiento del acuífero (9 SHAC)",
     "SHAC_storage_", "Mm³", 1e-6, "mean"),
    ("S2_deficit_ap", "Déficit de agua potable (8 localidades)",
     "AP_UnmetDemand__", "Mm³/año", 604800e-6, "sum"),
    ("S3_produccion", "Producción agrícola (palto)",
     "AGR_AnnualCropProduction", "kt/año", 1e-6, "sum"),
    ("S4_suministro", "Suministro por enlaces de transmisión",
     "AP_TransmissionLinks__", "Mm³/año", 1e-6, "sum"),
    ("S5_nivel", "Profundidad al agua (pozos de monitoreo)",
     "WF_DepthToWater_m", "m", 1.0, "mean"),
    ("S6_salinidad", "Interfaz salina en pozos costeros",
     "WF_Zvalue", "m", 1.0, "mean"),
]


def anual(x: np.ndarray, modo: str) -> np.ndarray:
    """(T,) semanal -> (n_años,) con la agregacion que corresponda.

    `sum` para flujos (deficit, produccion, suministro) y `mean` para estados
    (almacenamiento, nivel, salinidad): sumar un estado no significa nada.
    """
    n = len(x) // WPY
    y = x[:n * WPY].reshape(n, WPY)
    return y.sum(axis=1) if modo == "sum" else y.mean(axis=1)


def predecir(art_dir: str, X_raw, fn_raw, tn_raw):
    """Rollout libre con los artefactos de una iteracion. Devuelve (T, n_targets)
    en el ESPACIO DE NOMBRES del modelo, mas sus nombres."""
    import importlib, os
    os.environ["DPS_DATA_DIR"] = str(Path(art_dir).resolve())
    for m in [m for m in list(sys.modules) if m.startswith("weap_dps")]:
        del sys.modules[m]
    cfg = importlib.import_module("weap_dps.config_weap")
    from weap_dps.pipe_simulation_weap import PipeWEAP
    pipe = PipeWEAP(template_path=cfg.ZARR_TEMPLATE_PATH)
    surr = pipe.surrogate
    Xn = normalizar_x(surr, list(pipe.feature_names), X_raw, fn_raw)
    with torch.no_grad():
        gw_n, sf_n = surr.model.model.forward_sequence(
            torch.tensor(Xn[None, ...], dtype=torch.float32), cfg.WARMUP_WEEKS)
    gw = surr.denormalize_y(gw_n[0].numpy(), kind="gw")
    sf = surr.denormalize_y(sf_n[0].numpy(), kind="surface")
    return (np.hstack([gw, sf]),
            list(surr.target_names_gw) + list(surr.target_names_surf))


def main() -> int:
    Z = zarr.open_group(ZARR, mode="r")
    fn_raw = list(Z.attrs["feature_names"])
    tn_raw = list(Z.attrs["target_names"])
    rid = np.asarray(Z["run_ids"][:]).astype(int)

    datos = {}
    for r in RUNS:
        k = int(np.where(rid == r)[0][0])
        X_raw = np.nan_to_num(Z["X"][k])
        Y_raw = np.nan_to_num(Z["Y"][k])
        d = {"WMS2Ma": (Y_raw, tn_raw)}
        for et, art in ART.items():
            d[et] = predecir(art, X_raw, fn_raw, tn_raw)
            print(f"  run {r} · {et}: {d[et][0].shape[1]} columnas")
        datos[r] = d

    OUT.mkdir(parents=True, exist_ok=True)
    for vid, titulo, pat, uni, f, modo in VARS:
        fig, axes = plt.subplots(1, len(RUNS), figsize=(15.5, 4.1), sharey=True)
        for ax, r in zip(np.atleast_1d(axes), RUNS):
            for et in ("WMS2Ma", "iter01", "iter02"):
                Y, nom = datos[r][et]
                cols = [i for i, n in enumerate(nom) if n.startswith(pat)]
                if not cols:
                    continue
                serie = np.maximum(Y[W0:, cols], 0).sum(axis=1) if modo == "sum" \
                    else Y[W0:, cols].mean(axis=1)
                s = anual(serie, modo) * f
                ax.plot(2027 + np.arange(len(s)), s, lw=2.0 if et == "WMS2Ma" else 1.5,
                        color=COL[et], label=et,
                        ls="-" if et == "WMS2Ma" else "--", alpha=1 if et == "WMS2Ma" else .85)
            ax.set_title(f"run {r}", fontsize=10, weight="bold")
            ax.set_xlabel("año", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        np.atleast_1d(axes)[0].set_ylabel(f"{uni}", fontsize=9)
        np.atleast_1d(axes)[0].legend(fontsize=9, frameon=False)
        fig.suptitle(f"{titulo}  ·  agregado {'anual' if modo == 'sum' else 'medio anual'}",
                     fontsize=12, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        p = OUT / f"{vid}_weap_vs_iteraciones.png"
        fig.savefig(p, dpi=155); plt.close(fig)
        print(f"figura: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
