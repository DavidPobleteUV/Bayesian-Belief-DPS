# -*- coding: utf-8 -*-
"""
figuras_weap_vs_iter.py — WMS2Ma contra las dos iteraciones del emulador.

Una figura por objetivo, sobre las MISMAS 49 corridas del frente re-simuladas en
el modelo de referencia. El eje x es siempre WMS2Ma —la verdad contra la que se
mide— y el eje y la predicción del emulador, de modo que la diagonal es el acierto
perfecto y la distancia vertical a ella es el error.

Por qué scatter y no barras de error: con 49 corridas, un resumen agregado esconde
si el error es un sesgo uniforme —la nube paralela a la diagonal, corregible con
un factor— o dispersión —la nube abierta, que no se corrige—. Esa distinción es
la que decide si un objetivo sirve para ordenar carteras.

Uso:
    python weap_dps/figuras_weap_vs_iter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("results/figuras")
C1, C2 = "#c0392b", "#2874a6"          # iter01 rojo, iter02 azul
GRIS = "#7f8c8d"

# (columna, etiqueta, unidad, factor de escala, admite error RELATIVO)
#
# J2, J51 y J52 NO lo admiten: su valor de referencia es ~0 en las politicas
# buenas, y contra ese cero el relativo explota. El flag va explicito y no
# autodetectado, porque autodetectarlo fallaba —los valores son pequenos pero no
# nulos, de modo que ninguna guarda por magnitud los atrapaba— y producia sesgos
# reportados del 358 %.
OBJ = [
    ("J1_gw_storage", "J1 · Almacenamiento del acuífero", "Mm³", 1e-6, True),
    ("J2_unmet_ap", "J2 · Déficit de agua potable", "Mm³", 1e-6, False),
    ("J3_agri_value", "J3 · Valor agrícola (VAN)", "MUSD", 1 / 980e6, True),
    ("J4_supply_cost", "J4 · Costo de suministro (VAN)", "MUSD", 1 / 980e6, True),
    ("J51_mean_town_fail", "J51 · Semanas en falla por pueblo", "semanas", 1.0, False),
    ("J52_worst_year_frac", "J52 · Déficit del peor año", "fracción", 1.0, False),
    ("J6_coastal_salinity", "J6 · Interfaz salina costera", "m", 1.0, True),
]


def cargar():
    a = pd.read_csv("results/comparacion_weap_mlp_por_run.csv",
                    encoding="utf-8-sig").set_index("run").sort_index()
    b = pd.read_csv("results/comparacion_weap_mlp_iter02_por_run.csv",
                    encoding="utf-8-sig").set_index("run").sort_index()
    comunes = a.index.intersection(b.index)
    return a.loc[comunes], b.loc[comunes]


def _panel(ax, w, y1, y2, titulo, unidad, usa_rel):
    lo = min(w.min(), y1.min(), y2.min())
    hi = max(w.max(), y1.max(), y2.max())
    m = 0.06 * (hi - lo) if hi > lo else 1.0
    lim = (lo - m, hi + m)

    ax.plot(lim, lim, color=GRIS, lw=1.2, ls="--", zorder=1)
    ax.scatter(w, y1, s=34, color=C1, alpha=.75, edgecolor="none",
               label="iteración 1", zorder=3)
    ax.scatter(w, y2, s=34, color=C2, alpha=.75, edgecolor="none",
               label="iteración 2", zorder=4, marker="^")

    # Sesgo mediano: signo y magnitud, que es lo que distingue un sesgo
    # —nube paralela a la diagonal, corregible— de dispersion —nube abierta—.
    # Se reporta en RELATIVO solo si la referencia no se acerca a cero; si lo
    # hace, el relativo explota y se informa en la unidad del propio objetivo.
    for y, c, dy in ((y1, C1, 0.10), (y2, C2, 0.04)):
        if usa_rel:
            v = 100 * np.median((y - w) / np.abs(w)); txt = f"sesgo {v:+.1f} %"
        else:
            v = np.median(y - w); txt = f"sesgo {v:+.3g} {unidad}"
        ax.text(0.03, 0.97 - dy, txt, transform=ax.transAxes,
                color=c, fontsize=8.5, va="top", weight="bold")

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(titulo, fontsize=10, weight="bold")
    ax.set_xlabel(f"WMS2Ma  [{unidad}]", fontsize=8.5)
    ax.set_ylabel(f"emulador  [{unidad}]", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    a, b = cargar()
    print(f"corridas comparadas: {len(a)}")

    n = len(OBJ)
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.4))
    for k, (col, tit, uni, f, rel) in enumerate(OBJ):
        ax = axes.flat[k]
        w = a[f"weap_{col}"].to_numpy() * f
        y1 = a[f"mlp_{col}"].to_numpy() * f
        y2 = b[f"mlp_{col}"].to_numpy() * f
        _panel(ax, w, y1, y2, tit, uni, rel)
    axes.flat[n].axis("off")
    axes.flat[n].legend(*axes.flat[0].get_legend_handles_labels(),
                        loc="center", fontsize=11, frameon=False,
                        title="La línea punteada es el acierto perfecto.\n"
                              "Sesgo: mediana de (emulador menos WMS2Ma), "
                              "relativo salvo en J2, J51 y J52,\n"
                              "cuya referencia puede ser cero.",
                        title_fontsize=9)
    fig.suptitle("Emulador contra el modelo de referencia, por objetivo de decisión\n"
                 f"{len(a)} corridas del frente re-simuladas en WMS2Ma",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "V1_weap_vs_iteraciones.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  figura: {p}")

    # ── tabla de apoyo ──────────────────────────────────────────────────────
    # Se LEE de los resumenes que produce comparar_weap_mlp.py en vez de
    # recalcular el error aqui. Reimplementarlo fue justamente lo que introdujo
    # una metrica inconsistente: el relativo sobre J2 daba 358 %.
    r = []
    for et, f in (("iter01", "results/comparacion_weap_mlp.csv"),
                  ("iter02", "results/comparacion_weap_mlp_iter02.csv")):
        t = pd.read_csv(f, encoding="utf-8-sig")
        t.insert(0, "iteracion", et)
        r.append(t)
    t = pd.concat(r, ignore_index=True)
    t.to_csv(OUT.parent / "comparacion_iteraciones.csv", index=False,
             encoding="utf-8-sig")
    piv = t.pivot(index=["objetivo", "metrica"], columns="iteracion",
                  values=["mediana", "pct_dentro_tol"])
    print()
    print("error mediano (%):")
    print((100 * piv["mediana"]).round(2).to_string())
    print()
    print("dentro de tolerancia (%):")
    print(piv["pct_dentro_tol"].round(0).to_string())
    print()
    print(f"  tabla: {OUT.parent / 'comparacion_iteraciones.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
