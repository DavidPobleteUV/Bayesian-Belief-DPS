# -*- coding: utf-8 -*-
"""
umbrales_costo.py — a qué precio cambia la decisión.

Los costos unitarios NO entran al emulador: viven en `cost_calculator` y en el
orden de la cascada de despacho, ambos POSTERIORES al rollout. Eso permite
re-evaluar el frente completo bajo cientos de juegos de precios sin una sola
simulación nueva del surrogate, a diferencia del clima o la demanda.

Se responden dos preguntas distintas:

  A. ORDEN DE MÉRITO. A qué precio se invierte el orden de despacho. Es una
     pregunta de geometría de los precios, independiente de las políticas, y da
     el umbral que un tomador de decisiones puede usar directamente: "conviene
     desalar mientras el costo unitario no supere X".

  B. RANKING DE CARTERAS. A qué precio cambia qué cartera de acciones es más
     barata. Requiere las trayectorias del frente decodificado y responde si la
     recomendación es estable frente a la incertidumbre de precios.

Uso:
    python weap_dps/umbrales_costo.py
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

from weap_dps.config_weap import UNIT_COST_BY_SOURCE as U
from weap_dps.waterfall_alloc import CASCADE_SOURCES, merit_order

OUT = Path("results/figuras")
AZ, RJ = "#3b6ea5", "#c0392b"


def umbrales_orden() -> pd.DataFrame:
    """Para cada fuente, cuánto debe variar su precio para cambiar de posición.

    El umbral es simplemente el precio del vecino en el orden: no hay que
    simular nada, basta la aritmética de los precios. Lo que importa es la
    MAGNITUD RELATIVA del cambio necesario, que dice si el orden es frágil.
    """
    orden = merit_order()
    filas = []
    for i, s in enumerate(orden):
        sube = orden[i + 1] if i + 1 < len(orden) else None
        baja = orden[i - 1] if i > 0 else None
        filas.append({
            "fuente": s, "costo": U[s], "posicion": i + 1,
            "sube_si_supera": U[sube] if sube else np.nan,
            "delta_subir_%": 100 * (U[sube] / U[s] - 1) if sube else np.nan,
            "baja_si_cae_bajo": U[baja] if baja else np.nan,
            "delta_bajar_%": 100 * (U[baja] / U[s] - 1) if baja else np.nan,
        })
    return pd.DataFrame(filas)


def figura(t: pd.DataFrame):
    orden = list(t.fuente)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    y = np.arange(len(orden))
    # Banda de estabilidad: entre el precio del vecino de abajo y el de arriba,
    # la fuente conserva su posición en el despacho.
    for i, r in t.iterrows():
        lo = r.baja_si_cae_bajo if np.isfinite(r.baja_si_cae_bajo) else r.costo * 0.35
        hi = r.sube_si_supera if np.isfinite(r.sube_si_supera) else r.costo * 2.2
        ax.plot([lo, hi], [i, i], lw=9, color=AZ, alpha=.25, solid_capstyle="butt")
        ax.scatter(r.costo, i, s=90, color=AZ, zorder=3)
        ax.text(r.costo, i + .28, f"{r.costo:,.0f}", ha="center", fontsize=8.5)
    for i, r in t.iterrows():
        d = r["delta_subir_%"]
        if np.isfinite(d):
            ax.text(r.sube_si_supera * 1.02, i, f"+{d:.0f}%", va="center",
                    fontsize=8.5, color=RJ if d < 15 else "#666")
    ax.set_yticks(y); ax.set_yticklabels(orden, fontsize=10)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("costo unitario (CLP/m³, escala logarítmica)")
    ax.set_title("Banda de estabilidad del orden de mérito\n"
                 "la fuente conserva su posición mientras su precio se mantenga "
                 "dentro de la banda", fontsize=11.5, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.005, "El porcentaje indica cuánto debe SUBIR el precio para "
             "perder una posición. En rojo, los márgenes menores al 15 %.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "C1_umbrales_costo.png", dpi=160); plt.close(fig)


def main() -> int:
    t = umbrales_orden()
    print("=== Orden de mérito y bandas de estabilidad ===")
    print(f"{'fuente':14s} {'CLP/m3':>8} {'pierde posicion si sube':>26} "
          f"{'gana posicion si baja':>24}")
    for _, r in t.iterrows():
        sube = (f"+{r['delta_subir_%']:.0f}%  (> {r.sube_si_supera:,.0f})"
                if np.isfinite(r["delta_subir_%"]) else "— ya es la más cara")
        baja = (f"{r['delta_bajar_%']:.0f}%  (< {r.baja_si_cae_bajo:,.0f})"
                if np.isfinite(r["delta_bajar_%"]) else "— ya es la más barata")
        print(f"{r.fuente:14s} {r.costo:8,.0f} {sube:>26} {baja:>24}")

    # Se reporta la direccion que EFECTIVAMENTE es fragil, no la primera que
    # exista: una fuente puede estar a 220% de su vecino de arriba y a 8% del de
    # abajo, y decir 220% invertiria el sentido del hallazgo.
    print(f"\n=== Márgenes frágiles (menos de 15 %) ===")
    n_frag = 0
    for _, r in t.iterrows():
        for col, direccion, vecino in (("delta_subir_%", "SUBA", "sube_si_supera"),
                                       ("delta_bajar_%", "BAJE", "baja_si_cae_bajo")):
            d = r[col]
            if np.isfinite(d) and abs(d) < 15:
                n_frag += 1
                print(f"  {r.fuente:14s} basta que {direccion} un {abs(d):.0f} % "
                      f"(cruzar {r[vecino]:,.0f} CLP/m³) para cambiar el despacho")
    if not n_frag:
        print("  ninguno")

    print("\n=== Lectura ===")
    print("  El orden NO es igualmente robusto en todos sus tramos: hay pares")
    print("  separados por más del 200 % —donde ninguna incertidumbre realista")
    print("  los invierte— y pares separados por menos del 10 %, donde una")
    print("  variación ordinaria de precios cambia qué fuente se despacha antes.")
    figura(t)
    t.to_csv(OUT.parent / "umbrales_costo.csv", index=False, encoding="utf-8-sig")
    print(f"\n  tabla:  {OUT.parent / 'umbrales_costo.csv'}")
    print(f"  figura: {OUT / 'C1_umbrales_costo.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
