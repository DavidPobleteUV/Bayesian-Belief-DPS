# -*- coding: utf-8 -*-
"""
series_gw_baseline.py — series de tiempo del almacenamiento subterráneo,
WMMaS2 contra el emulador, en runs SIN ninguna acción de adaptación.

Por qué runs sin acciones: aíslan la dinámica hidrológica del efecto de las
decisiones. Lo que se ve es si el emulador reproduce la evolución del acuífero
bajo clima y demanda solamente, que es la base sobre la que después se apilan
las acciones.

Los runs se toman del conjunto de TEST, de modo que el modelo no los vio durante
el entrenamiento, y se recorren en rollout recursivo LIBRE: tras el calentamiento
de 104 semanas el emulador se alimenta de sus propias predicciones, sin ninguna
corrección externa a lo largo de los 44 años restantes.

Uso:
    python weap_dps/series_gw_baseline.py --runs 1 12 26 37
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weap_dps.comparar_weap_mlp import kge, normalizar_x
from weap_dps.config_weap import WARMUP_WEEKS, ZARR_TEMPLATE_PATH

ZARR = Path(r"C:\Users\David\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW"
            r"\data\_v3_900_fix2050\weap_weekly_merged.zarr")
MASTER = Path(r"C:\Users\David\Documents\GitHub_DPL\WEAP_2_ZARR\data\RunIDs_Q_full.csv")
# Fidelidad del emulador: la salida pertenece al repo de entrenamiento.
OUT = Path(__file__).resolve().parents[2] / "WEAP_HydroMLP_RecursiveGW" / "results" / "iter1_fix2050"
ANIO0 = 2014


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 12, 26, 37])
    ap.add_argument("--out", type=Path, default=OUT / "S1_gw_baseline.png")
    args = ap.parse_args()

    torch.set_num_threads(1); torch.set_grad_enabled(False)
    Z = zarr.open_group(str(ZARR), mode="r")
    rid = np.asarray(Z["run_ids"][:]).astype(int)
    fn = list(Z.attrs["feature_names"]); tn = list(Z.attrs["target_names"])
    m = pd.read_csv(MASTER, encoding="utf-8-sig").set_index("ID")

    from weap_dps.main_robust_weap import RobustPipeWEAP
    pipe = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=1.0)
    surr = pipe.surrogate
    # Almacenamiento por acuífero: es la variable de J1 y la de dinámica más lenta.
    j_alm = [j for j, n in enumerate(surr.target_names_gw)
             if n.startswith("SHAC_storage_")]
    ti = {n: i for i, n in enumerate(tn)}
    print(f"acuíferos con almacenamiento: {len(j_alm)}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6), sharex=True)
    t = ANIO0 + np.arange(2392) / 52.0
    for ax, r in zip(axes.ravel(), args.runs):
        k = int(np.where(rid == r)[0][0])
        X_raw, Y_raw = np.nan_to_num(Z["X"][k]), np.nan_to_num(Z["Y"][k])
        Xn = normalizar_x(surr, list(pipe.feature_names), X_raw, fn)
        gw_n, _ = surr.model.model.forward_sequence(
            torch.tensor(Xn[None, ...], dtype=torch.float32), WARMUP_WEEKS)
        gw = surr.denormalize_y(gw_n[0].numpy(), kind="gw")
        gw_w = np.stack([Y_raw[:, ti[n]] for n in surr.target_names_gw], axis=1)

        sw = gw_w[:, j_alm].sum(axis=1) / 1e6          # Mm3
        sp = gw[:, j_alm].sum(axis=1) / 1e6
        k_post = kge(sw[WARMUP_WEEKS:], sp[WARMUP_WEEKS:])

        # Se grafica DESDE el fin del calentamiento. Durante esas 104 semanas el
        # emulador aun se alimenta de rezagos observados y su salida no es
        # comparable: produce un transitorio de arranque que no forma parte de
        # la evaluacion (el KGE tambien se calcula post-calentamiento).
        w = WARMUP_WEEKS
        ax.plot(t[w:], sw[w:], color="black", lw=1.5, label="WMMaS2", zorder=3)
        ax.plot(t[w:], sp[w:], color="#3b6ea5", lw=1.3, alpha=0.9,
                label="Emulador", zorder=4)
        row = m.loc[r]
        area = str(row.Demanda_Agro).replace(" Areas Regadas", " sup. regada")
        pob = str(row.Demanda_Poblacion).split(":")[-1].strip()
        ax.set_title(f"run {r} · {row.GCM} {row.SSP}\n{area} · población {pob}"
                     f"   |   KGE = {k_post:.3f}", fontsize=9.5)
        ax.set_ylabel("almacenamiento total (Mm³)", fontsize=9)
        ax.legend(fontsize=8.5, frameon=False, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
        print(f"  run {r:4d}  KGE={k_post:.3f}  "
              f"WEAP {sw[WARMUP_WEEKS:].mean():.1f} vs MLP {sp[WARMUP_WEEKS:].mean():.1f} Mm3")
    for ax in axes[1]:
        ax.set_xlabel("año")
    fig.suptitle("Almacenamiento subterráneo — runs SIN acciones de adaptación "
                 "(conjunto de test, rollout libre)", fontsize=12.5, weight="bold")
    fig.text(0.5, 0.005, "Se muestra desde el fin del calentamiento (104 semanas): "
             "a partir de ahí el emulador se alimenta solo de sus propias "
             "predicciones durante 44 años.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160); plt.close(fig)
    print(f"\nfigura: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
