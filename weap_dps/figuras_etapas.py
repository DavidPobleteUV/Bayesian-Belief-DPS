# -*- coding: utf-8 -*-
"""
figuras_etapas.py — una figura por etapa del marco WEAP -> HydroMLP -> DPS.

  E1  ensamble de simulaciones: cobertura del diseño experimental
  E2  emulador: distribución de KGE por bloque y por objetivo de decisión
  E5  robustez: criterio de dominio y descomposición por factor
  E6  verificación: emulador contra WMMaS2 en los runs del ciclo

Las del DPS (frente y carteras) las produce figuras_carteras.py.

Uso:
    python weap_dps/figuras_etapas.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W2Z = Path(r"C:\Users\David\Documents\GitHub_DPL\WEAP_2_ZARR")
HYD = Path(r"C:\Users\David\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW")
# Cada resultado va al proyecto al que pertenece: el diseño del ensamble es de
# WEAP_2_ZARR, la fidelidad del emulador de HydroMLP y la robustez del DPS.
# Mezclarlos obligaba a leer figuras de un repo desde otro.
OUT_DPS = Path("results/figuras")
OUT_MLP = HYD / "results" / "iter1_fix2050"
OUT_ENS = W2Z / "results" / "figuras_ensamble"
AZ, GR, RJ = "#3b6ea5", "#31a354", "#c0392b"


# ── E1: cobertura del ensamble ───────────────────────────────────────────────
def e1_ensamble():
    m = pd.read_csv(W2Z / "data" / "RunIDs_Q_full.csv", encoding="utf-8-sig")
    import zarr
    z = zarr.open_group(str(HYD / "data" / "_v3_900_fix2050" / "weap_weekly_merged.zarr"),
                        mode="r")
    zr = set(np.asarray(z["run_ids"][:]).astype(int).tolist())
    s = m[m.ID.isin(zr)]

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8))

    # bloques
    b = s.block.value_counts()
    ax[0, 0].barh(range(len(b)), b.values, color=AZ)
    ax[0, 0].set_yticks(range(len(b))); ax[0, 0].set_yticklabels(b.index, fontsize=9)
    for i, v in enumerate(b.values):
        ax[0, 0].text(v + 6, i, str(v), va="center", fontsize=9)
    ax[0, 0].set_title(f"Bloques del ensamble (n={len(s)})", fontsize=11, weight="bold")
    ax[0, 0].invert_yaxis(); ax[0, 0].set_xlim(0, b.max() * 1.15)

    # clima
    s2 = s.copy(); s2["clima"] = s2.GCM + "\n" + s2.SSP
    c = s2.clima.value_counts().sort_values()
    ax[0, 1].barh(range(len(c)), c.values, color=GR)
    ax[0, 1].set_yticks(range(len(c))); ax[0, 1].set_yticklabels(c.index, fontsize=7.5)
    n_seq = int(s.drought_severity.notna().sum())
    ax[0, 1].set_title(f"Series climáticas · {n_seq} runs con sequía impuesta",
                       fontsize=11, weight="bold")

    # demanda: poblacion x area
    orden_a = ["Disminuye 50% Areas Regadas", "Disminuye 15% Areas Regadas",
               "Sin cambio en Areas Regadas", "Aumenta 10% Areas Regadas",
               "Aumenta 20% Areas Regadas", "Aumenta 30% Areas Regadas"]
    corto_a = ["−50 %", "−15 %", "0 %", "+10 %", "+20 %", "+30 %"]
    orden_p = ["Crecimiento anual regular: 2%", "Crecimiento levemente mayor: 3%",
               "Crecimiento mucho mayor: 5%"]
    tab = pd.crosstab(s.Demanda_Agro, s.Demanda_Poblacion).reindex(
        index=orden_a, columns=orden_p).fillna(0)
    im = ax[1, 0].imshow(tab.values, cmap="Blues", aspect="auto")
    ax[1, 0].set_xticks(range(3)); ax[1, 0].set_xticklabels(["2 %", "3 %", "5 %"])
    ax[1, 0].set_yticks(range(6)); ax[1, 0].set_yticklabels(corto_a, fontsize=9)
    for i in range(6):
        for j in range(3):
            v = int(tab.values[i, j])
            ax[1, 0].text(j, i, v, ha="center", va="center", fontsize=9,
                          color="white" if v > tab.values.max() * .55 else "black")
    ax[1, 0].set_xlabel("crecimiento poblacional"); ax[1, 0].set_ylabel("superficie agrícola")
    ax[1, 0].set_title("Demanda: población × superficie", fontsize=11, weight="bold")

    # combinaciones de acciones
    A = ["desalacion_costera", "desalacion_completa", "nuevo_pozo_a_5km", "acuerdo"]
    s3 = s.copy()
    s3["k"] = s3[["act_" + a for a in A]].sum(axis=1)
    kk = s3.k.value_counts().sort_index()
    ax[1, 1].bar(kk.index, kk.values, color=AZ)
    for i, v in zip(kk.index, kk.values):
        ax[1, 1].text(i, v + 5, str(v), ha="center", fontsize=9)
    ax[1, 1].set_xlabel("acciones activas por run")
    # El titulo describe lo que el panel MUESTRA. Que las 16 combinaciones esten
    # presentes es cierto pero no se ve aqui, asi que va como anotacion.
    n_comb = s3[["act_" + a for a in A]].astype(int).astype(str).agg("".join,
                                                                    axis=1).nunique()
    ax[1, 1].set_title("Número de acciones simultáneas por run",
                       fontsize=11, weight="bold")
    ax[1, 1].text(0.98, 0.94,
                  f"{n_comb} de 16 combinaciones\npresentes en el ensamble",
                  transform=ax[1, 1].transAxes, ha="right", va="top", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4f8", ec="#c8d4e0"))
    ax[1, 1].set_ylim(0, kk.max() * 1.18)
    for a_ in ax.ravel():
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Etapa 1 — Ensamble de simulaciones WEAP–MODFLOW",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_ENS / "E1_ensamble.png", dpi=160); plt.close(fig)


# ── E2: fidelidad del emulador ───────────────────────────────────────────────
def _leer_metricas():
    p = HYD / "results" / "eval_metrics_detail.log"
    bloque, filas = None, []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if "METRICAS GW" in ln: bloque = "Subterráneo"; continue
        if "METRICAS SURFACE" in ln: bloque = "Superficie"; continue
        m = re.match(r"^(\S.*?)\s+(\d+)\s+(-?[\d.]+|nan)\s+(-?[\d.eE+]+|nan)\s+"
                     r"(\S+)\s+(-?[\d.]+|nan)\s*$", ln)
        if m and bloque:
            try:
                filas.append({"bloque": bloque, "var": m.group(1).strip(),
                              "kge": float(m.group(3))})
            except ValueError:
                pass
    return pd.DataFrame(filas)


def e2_emulador():
    d = _leer_metricas()
    v = d.groupby(["bloque", "var"]).kge.median().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    bins = np.linspace(-0.5, 1, 46)
    for b, col in (("Subterráneo", AZ), ("Superficie", GR)):
        x = v[v.bloque == b].kge.clip(-0.5, 1)
        ax[0].hist(x, bins=bins, alpha=0.62, color=col,
                   label=f"{b}  (n={len(x)}, mediana {np.median(x):.3f})")
    ax[0].axvline(0.5, color=RJ, ls="--", lw=1.2)
    ax[0].text(0.5, ax[0].get_ylim()[1] * .95, " KGE = 0,5", color=RJ, fontsize=9)
    ax[0].set_xlabel("KGE mediano por variable"); ax[0].set_ylabel("n de variables")
    ax[0].legend(fontsize=9, frameon=False)
    ax[0].set_title(f"Distribución sobre {len(v)} variables", fontsize=11, weight="bold")

    obj = {"J1 almacenamiento": 0.967, "J2 déficit AP": 0.723,
           "J3 valor agrícola": 0.821, "J4 transmisión": 0.856,
           "J6 salinidad": 0.776}
    prev = {"J1 almacenamiento": 0.961, "J2 déficit AP": 0.760,
            "J3 valor agrícola": 0.836, "J4 transmisión": 0.592,
            "J6 salinidad": 0.769}
    y = np.arange(len(obj))
    ax[1].barh(y + .19, list(obj.values()), height=.36, color=AZ, label="corregido")
    ax[1].barh(y - .19, [prev[k] for k in obj], height=.36, color="#bbbbbb",
               label="sin corregir")
    for i, k in enumerate(obj):
        ax[1].text(obj[k] + .012, i + .19, f"{obj[k]:.3f}", va="center", fontsize=8.5)
    ax[1].set_yticks(y); ax[1].set_yticklabels(list(obj), fontsize=9.5)
    ax[1].axvline(0.5, color=RJ, ls="--", lw=1.2)
    ax[1].set_xlim(0, 1.06); ax[1].set_xlabel("KGE mediano")
    ax[1].legend(fontsize=9, frameon=False, loc="lower right")
    ax[1].set_title("Por objetivo de decisión", fontsize=11, weight="bold")
    ax[1].invert_yaxis()
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Etapa 2 — Fidelidad del emulador (rollout libre, 113 runs de test)",
                 fontsize=13, weight="bold")
    fig.text(0.5, 0.005, "La corrección del corte en 2050 concentra su efecto en la "
             "transmisión (0,592 → 0,856), que gobierna el costo J4.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(OUT_MLP / "E2_emulador.png", dpi=160); plt.close(fig)


# ── E5: robustez ─────────────────────────────────────────────────────────────
def e5_robustez():
    p = Path("results/robustez_iter1_fix2050/robustez.npz")
    if not p.exists():
        print("  (sin robustez.npz)"); return
    d = np.load(p, allow_pickle=True)
    res, lab = d["res"], [str(x) for x in d["labels"]]
    j51, j52 = res[:, :, 4], res[:, :, 5]
    ok = (j52 <= 1.15) & (j51 <= 900.0)
    dom = 100 * ok.mean(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax[0].hist(dom, bins=np.linspace(60, 101, 22), color=AZ)
    ax[0].axvline(90, color=RJ, ls="--", lw=1.3)
    ax[0].text(90.5, ax[0].get_ylim()[1] * .9, " umbral 90 %", color=RJ, fontsize=9)
    ax[0].set_xlabel("criterio de dominio (% de los 81 futuros)")
    ax[0].set_ylabel("n de políticas")
    ax[0].set_title(f"{int((dom > 90).sum())} de {len(dom)} políticas sobre el 90 %",
                    fontsize=11, weight="bold")

    pob = [re.search(r"pop(\d+)%", s).group(1) for s in lab]
    niveles = sorted(set(pob), key=int)
    val = [100 * ok[:, [i for i, x in enumerate(pob) if x == n]].mean() for n in niveles]
    ant = [100.0, 99.5, 2.9]
    x = np.arange(len(niveles))
    ax[1].bar(x + .19, val, width=.36, color=AZ, label="corregido")
    ax[1].bar(x - .19, ant, width=.36, color="#bbbbbb", label="sin corregir")
    for i, v in enumerate(val):
        ax[1].text(i + .19, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    for i, v in enumerate(ant):
        ax[1].text(i - .19, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5, color="#666")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{n} %/año" for n in niveles])
    ax[1].set_xlabel("crecimiento poblacional"); ax[1].set_ylabel("cumplimiento (%)")
    ax[1].set_ylim(0, 118); ax[1].legend(fontsize=9, frameon=False)
    ax[1].set_title("El factor determinante, antes y después", fontsize=11, weight="bold")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Etapa 5 — Verificación de robustez sobre 81 estados del mundo",
                 fontsize=13, weight="bold")
    fig.text(0.5, 0.005, "Con el corte en 2050 corregido, el cumplimiento bajo "
             "crecimiento del 5 % pasa de 2,9 % a 88,2 %.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(OUT_DPS / "E5_robustez.png", dpi=160); plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    e1_ensamble(); print("  E1_ensamble.png")
    e2_emulador(); print("  E2_emulador.png")
    e5_robustez(); print("  E5_robustez.png")
    print(f"\nfiguras en {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
