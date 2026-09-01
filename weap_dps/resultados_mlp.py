# -*- coding: utf-8 -*-
"""
resultados_mlp.py — paquete de resultados del emulador: tablas y series.

Genera, en la carpeta de resultados del repo de ENTRENAMIENTO (que es a donde
pertenecen por la frontera entre repos):

  kge_por_familia.csv     KGE por familia de variables, con n, mediana, IQR y PBIAS
  M1_kge_familias.png     la misma tabla como figura, separada por bloque
  M2_series_superficie.png series de producción agrícola y déficit de agua potable
  M3_series_gw.png        series de almacenamiento subterráneo (ver también S1)

NOTA DE UBICACION. El script vive en el repo del DPS porque reutiliza su cargador
del surrogate (MLPSurrogate + template). La evaluación del emulador pertenece
conceptualmente a WEAP_HydroMLP_RecursiveGW, de modo que las SALIDAS se escriben
allí. Moverlo del todo exige portar el cargador; queda anotado.

Uso:
    python weap_dps/resultados_mlp.py --runs 1 12 26 37
"""
from __future__ import annotations

import argparse
import re
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

HYD = Path(r"C:\Users\David\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW")
ZARR = HYD / "data" / "_v3_900_fix2050" / "weap_weekly_merged.zarr"
LOG = HYD / "results" / "eval_metrics_detail.log"
OUT = HYD / "results" / "iter1_fix2050"
MASTER = Path(r"C:\Users\David\Documents\GitHub_DPL\WEAP_2_ZARR\data\RunIDs_Q_full.csv")
ANIO0, AZ, GR, RJ = 2014, "#3b6ea5", "#31a354", "#c0392b"

# Familias legibles. El orden define el de la figura.
BONITO = {
    "SHAC_storage": "Almacenamiento por acuífero",
    "WF_DepthToWater_m": "Profundidad al nivel freático",
    "SHAC_wells_lateralrecharge": "Recarga lateral",
    "SHAC_intershac": "Flujo inter-acuífero",
    "SHAC_drains": "Drenes",
    "SHAC_recharge": "Recarga areal",
    "SHAC_wells_demagro_returnflow": "Retorno de riego",
    "SHAC_wells_demagro_extraction": "Extracción agrícola",
    "SHAC_wells_demotras_extraction": "Extracción, otros usos",
    "WF_Zvalue": "Interfaz salina",
    "ETPotential": "Evapotranspiración potencial",
    "ETActual": "Evapotranspiración real",
    "AGR_AnnualCropProduction": "Producción agrícola anual",
    "AGR_UnmetDemand": "Déficit agrícola",
    "AGR_DailyIrrigation_m3": "Riego diario",
    "AP_UnmetDemand": "Déficit de agua potable",
    "AP_TransmissionLinks": "Enlaces de transmisión",
    "Caudales_rio": "Caudales del río",
}


def familia(nombre: str) -> str:
    for k, v in BONITO.items():
        if nombre.startswith(k):
            return v
    return nombre.split("__")[0][:30]


# ── tabla de KGE por familia ────────────────────────────────────────────────
def tabla_kge() -> pd.DataFrame:
    bloque, filas = None, []
    for ln in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if "METRICAS GW" in ln: bloque = "Subterráneo"; continue
        if "METRICAS SURFACE" in ln: bloque = "Superficie"; continue
        m = re.match(r"^(\S.*?)\s+(\d+)\s+(-?[\d.]+|nan)\s+(-?[\d.eE+]+|nan)\s+"
                     r"(\S+)\s+(-?[\d.]+|nan)\s*$", ln)
        if m and bloque:
            try:
                filas.append({"bloque": bloque, "var": m.group(1).strip(),
                              "kge": float(m.group(3)), "pbias": float(m.group(6))})
            except ValueError:
                pass
    d = pd.DataFrame(filas)
    d["familia"] = d["var"].map(familia)
    # Mediana por VARIABLE primero (entre runs), luego se agrega por familia:
    # promediar sobre pares variable-run daría más peso a las familias con más
    # variables dentro de cada run.
    v = d.groupby(["bloque", "familia", "var"]).agg(
        kge=("kge", "median"), pbias=("pbias", "median")).reset_index()
    t = v.groupby(["bloque", "familia"]).agg(
        n_variables=("var", "count"),
        kge_mediana=("kge", "median"),
        kge_p25=("kge", lambda s: s.quantile(.25)),
        kge_p75=("kge", lambda s: s.quantile(.75)),
        pbias_mediana=("pbias", "median")).reset_index()
    return t.sort_values(["bloque", "kge_mediana"], ascending=[True, False])


def fig_kge(t: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2),
                             gridspec_kw={"width_ratios": [1, 1]})
    for ax, (b, col) in zip(axes, (("Subterráneo", AZ), ("Superficie", GR))):
        s = t[t.bloque == b].sort_values("kge_mediana")
        y = np.arange(len(s))
        ax.hlines(y, s.kge_p25, s.kge_p75, color=col, lw=5, alpha=.35)
        ax.scatter(s.kge_mediana, y, s=64, color=col, zorder=3)
        for i, (k_, n_) in enumerate(zip(s.kge_mediana, s.n_variables)):
            ax.text(min(k_, 1.0) + .03, i, f"{k_:.3f}  (n={n_})", va="center",
                    fontsize=8.5)
        ax.set_yticks(y); ax.set_yticklabels(s.familia, fontsize=9)
        ax.axvline(0.5, color=RJ, ls="--", lw=1.1)
        ax.set_xlim(-0.7, 1.45); ax.set_xlabel("KGE (mediana y rango intercuartil)")
        ax.set_title(f"{b}  —  {int(s.n_variables.sum())} variables",
                     fontsize=11, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Emulador: KGE por familia de variables "
                 "(rollout libre, 113 runs de test)", fontsize=13, weight="bold")
    fig.text(0.5, 0.005, "Punto: mediana entre variables de la familia. Barra: "
             "rango intercuartil. La línea roja marca KGE = 0,5.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(OUT / "M1_kge_familias.png", dpi=160); plt.close(fig)


# ── series de superficie ────────────────────────────────────────────────────
def series(runs: list[int], anual: bool = False):
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    Z = zarr.open_group(str(ZARR), mode="r")
    rid = np.asarray(Z["run_ids"][:]).astype(int)
    fn, tn = list(Z.attrs["feature_names"]), list(Z.attrs["target_names"])
    m = pd.read_csv(MASTER, encoding="utf-8-sig").set_index("ID")
    from weap_dps.main_robust_weap import RobustPipeWEAP
    pipe = RobustPipeWEAP(template_path=ZARR_TEMPLATE_PATH, lam=1.0)
    surr = pipe.surrogate
    ti = {n: i for i, n in enumerate(tn)}
    ns = surr.target_names_surf
    j_prod = [j for j, n in enumerate(ns) if n.startswith("AGR_AnnualCropProduction")]
    j_unm = [j for j, n in enumerate(ns) if n.startswith("AP_UnmetDemand")]
    print(f"producción agrícola: {len(j_prod)} vars · déficit AP: {len(j_unm)} vars")

    t = ANIO0 + np.arange(2392) / 52.0
    w = WARMUP_WEEKS
    fig, axes = plt.subplots(2, len(runs), figsize=(3.9 * len(runs), 7),
                             sharex=True)
    for c, r in enumerate(runs):
        k = int(np.where(rid == r)[0][0])
        X_raw, Y_raw = np.nan_to_num(Z["X"][k]), np.nan_to_num(Z["Y"][k])
        Xn = normalizar_x(surr, list(pipe.feature_names), X_raw, fn)
        _, sf_n = surr.model.model.forward_sequence(
            torch.tensor(Xn[None, ...], dtype=torch.float32), w)
        sf = surr.denormalize_y(sf_n[0].numpy(), kind="surface")
        sf_w = np.stack([Y_raw[:, ti[n]] for n in ns], axis=1)

        for fila, (jj, nom, esc, uni) in enumerate((
                (j_prod, "Producción agrícola", 1e3, "ton/año"),
                (j_unm, "Déficit de agua potable", 1 / 604800 * 1e6, "Mm³/sem"))):
            ax = axes[fila, c] if len(runs) > 1 else axes[fila]
            vw = sf_w[:, jj].sum(axis=1)
            vp = sf[:, jj].sum(axis=1)
            if fila == 1:                       # déficit viene en m³/s
                vw, vp = vw * 604800 / 1e6, vp * 604800 / 1e6
            else:
                vw, vp = vw / 1e3, vp / 1e3

            if anual:
                # Agregacion natural de cada variable, no una media indiscriminada:
                # la produccion ya es una TASA anual que WEAP mantiene constante
                # dentro del ano (promediar recupera ese valor y elimina el ruido
                # semanal del emulador); el deficit es un VOLUMEN semanal, cuyo
                # agregado con sentido fisico es la suma del ano.
                n_y = (len(vw) - w) // 52
                ejex = ANIO0 + w / 52 + np.arange(n_y)
                red = (lambda a: a[w:w + n_y * 52].reshape(n_y, 52).mean(axis=1))                     if fila == 0 else                     (lambda a: a[w:w + n_y * 52].reshape(n_y, 52).sum(axis=1))
                yw, yp = red(vw), red(vp)
                kk = kge(yw, yp)
                ax.plot(ejex, yw, color="black", lw=1.6, marker="o", ms=2.6,
                        label="WMMaS2")
                ax.plot(ejex, yp, color=AZ if fila == 0 else GR, lw=1.4,
                        marker="o", ms=2.6, alpha=.9, label="Emulador")
            else:
                kk = kge(vw[w:], vp[w:])
                ax.plot(t[w:], vw[w:], color="black", lw=1.3, label="WMMaS2")
                ax.plot(t[w:], vp[w:], color=AZ if fila == 0 else GR, lw=1.1,
                        alpha=.9, label="Emulador")
            if fila == 0:
                ax.set_title(f"run {r} · {m.loc[r].GCM}", fontsize=9.5)
            # La unidad del déficit cambia con la agregación: semanal es un
            # caudal por semana, anual es el volumen acumulado del año.
            uni_txt = ("miles ton/año" if fila == 0
                       else ("Mm³/año" if anual else "Mm³/sem"))
            ax.set_ylabel(f"{nom}\n({uni_txt})",
                          fontsize=8.5)
            ax.text(.02, .93, f"KGE = {kk:.3f}", transform=ax.transAxes,
                    fontsize=8.5, va="top")
            if c == 0 and fila == 0:
                ax.legend(fontsize=8, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            print(f"  run {r:4d}  {nom:26s} KGE={kk:6.3f}")
        (axes[1, c] if len(runs) > 1 else axes[1]).set_xlabel("año")
    suf = " — agregado anual" if anual else ""
    fig.suptitle(f"Emulador contra WMMaS2 — variables de superficie{suf}",
                 fontsize=12.5, weight="bold")
    pie = ("Desde el fin del calentamiento. Producción: media del año. "
           "Déficit: volumen acumulado del año." if anual else
           "Desde el fin del calentamiento. La producción agrícola "
           "es anual (escalones); el déficit es semanal e intermitente.")
    fig.text(0.5, 0.005, pie, ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    nom_out = "M2b_series_superficie_anual.png" if anual else "M2_series_superficie.png"
    fig.savefig(OUT / nom_out, dpi=160); plt.close(fig)
    return nom_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 12, 26, 37])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t = tabla_kge()
    t.to_csv(OUT / "kge_por_familia.csv", index=False, encoding="utf-8-sig")
    print(f"tabla: {OUT / 'kge_por_familia.csv'}  ({len(t)} familias)")
    fig_kge(t); print("  M1_kge_familias.png")
    print("\n[resolución semanal]")
    print("  " + series(args.runs, anual=False))
    print("\n[agregado anual]")
    print("  " + series(args.runs, anual=True))
    print(f"\ntodo en {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
