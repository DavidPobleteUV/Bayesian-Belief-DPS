# -*- coding: utf-8 -*-
"""
analyze_pareto.py — Convergencia y estructura del frente robusto.

Responde dos preguntas distintas:

  1. ¿Convergió la optimización?  Semillas independientes que producen frentes
     equivalentes indican que el presupuesto de evaluaciones alcanzó. Se mide
     con hipervolumen (dispersión entre semillas) y epsilon-aditivo (cuánto le
     falta a cada semilla para cubrir el frente combinado).

  2. ¿Qué objetivos DISCRIMINAN?  Un objetivo casi constante a lo largo del
     frente no aporta trade-off: solo infla la dimensión y, con ella, la
     fracción de soluciones no dominadas por puro efecto de dimensionalidad.

Detecta 6 o 7 objetivos automáticamente (J5 se partió en J51/J52).

Uso:
    python weap_dps/analyze_pareto.py results/robust_iter1_h256
    python weap_dps/analyze_pareto.py <dir> --fig frente.png
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# Nombre legible y signo de cada objetivo. J1 y J3 se optimizan NEGADOS (NSGA
# minimiza todo), así que para mostrarlos hay que devolverles el signo.
PRETTY = {
    "J1_gw_storage":       ("GW storage", -1),
    "J2_unmet_ap":         ("Déficit AP", 1),
    "J3_agri_value":       ("Valor agrícola", -1),
    "J4_supply_cost":      ("Costo J4", 1),
    "J5_weeks_failure":    ("Semanas falla", 1),
    "J51_mean_town_fail":  ("Sem. falla/pueblo", 1),
    "J52_worst_year_frac": ("Peor año", 1),
    "J6_coastal_salinity": ("Salinidad", 1),
}
# Corridas viejas no guardaban los nombres: se deducen de cuántos objetivos hay.
LEGACY = {
    6: ["J1_gw_storage", "J2_unmet_ap", "J3_agri_value", "J4_supply_cost",
        "J5_weeks_failure", "J6_coastal_salinity"],
    7: ["J1_gw_storage", "J2_unmet_ap", "J3_agri_value", "J4_supply_cost",
        "J51_mean_town_fail", "J52_worst_year_frac", "J6_coastal_salinity"],
}


def nondominated(F: np.ndarray) -> np.ndarray:
    """Índices del frente no dominado (convención: todo se minimiza)."""
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if keep[i] and (np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)).any():
            keep[i] = False
    return np.where(keep)[0]


def hv_mc(P: np.ndarray, ref: float = 1.1, n: int = 200_000, seed: int = 0) -> float:
    """Hipervolumen por Monte Carlo en [0, ref]^d.

    Exacto es exponencial en la dimensión; con 6-7 objetivos y 100 puntos el
    estimador Monte Carlo basta para COMPARAR semillas, que es el uso acá.
    """
    rs = np.random.default_rng(seed)
    pts = rs.random((n, P.shape[1])) * ref
    dom = np.zeros(n, dtype=bool)
    for p in P:
        dom |= np.all(pts >= p, axis=1)
    return float(dom.mean() * ref ** P.shape[1])


def eps_add(A: np.ndarray, R: np.ndarray) -> float:
    """Epsilon-aditivo: cuánto hay que empeorar R para que A lo cubra."""
    return float(max(min(max(a[i] - r[i] for i in range(len(r))) for a in A) for r in R))


def load(d: Path):
    """F = objetivos OPTIMIZADOS (los que vio NSGA). A = los 7, si están.

    Se separan a propósito: la convergencia se mide en el espacio que la
    búsqueda realmente exploró, mientras que los diagnósticos (J1, J6) solo
    se reportan. Mezclarlos falsearía el hipervolumen.
    """
    seeds, F, V, A, el = [], {}, {}, {}, {}
    names = diag = None
    for f in sorted(d.glob("pareto_seed*.dat")):
        s = int(f.stem.replace("pareto_seed", ""))
        obj = pickle.load(open(f, "rb"))
        F[s] = np.array([r[1] for r in obj["result"]], dtype=float)
        V[s] = np.array([r[0] for r in obj["result"]], dtype=float)
        el[s] = obj.get("elapsed", float("nan")) / 3600.0
        ao = obj.get("all_objectives")
        if ao and all(x is not None for x in ao):
            A[s] = np.array(ao, dtype=float)
        names = obj.get("objective_names", names)
        diag = obj.get("objectives_diagnostic", diag)
        opt = obj.get("objectives_optimized")
        seeds.append(s)
    if not seeds:
        raise SystemExit(f"No hay pareto_seed*.dat en {d}")
    n = F[seeds[0]].shape[1]
    if names is None:                       # corrida antigua sin metadatos
        names = LEGACY.get(n)
        opt = names
        diag = []
        if names is None:
            raise SystemExit(f"{n} objetivos y sin 'objective_names' en el .dat")
    return seeds, F, V, A, el, names, opt or names, diag or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--fig", type=Path, default=None, help="PNG del frente")
    a = ap.parse_args()

    seeds, F, V, A, el, names, opt, diag = load(a.dir)
    spec = [PRETTY.get(o, (o, 1)) for o in opt]

    print(f"optimizados ({len(opt)}): {', '.join(opt)}")
    if diag:
        print(f"diagnóstico ({len(diag)}): {', '.join(diag)}")
    for s in seeds:
        print(f"seed {s:5d}: {F[s].shape[0]:3d} soluciones | "
              f"{V[s].shape[1]:3d} params | {el[s]:5.1f} h")

    allF = np.vstack([F[s] for s in seeds])
    owner = np.concatenate([[s] * F[s].shape[0] for s in seeds])
    lo, hi = allF.min(0), allF.max(0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    allN = (allF - lo) / rng
    ref_idx = nondominated(allN)
    ref = allN[ref_idx]

    print(f"\ntotal {len(allF)} soluciones | frente combinado: {len(ref)} "
          f"({100*len(ref)/len(allF):.0f}% no dominadas)")

    print(f"\n=== ¿discrimina cada objetivo? (rango sobre el frente) ===")
    print(f"{'objetivo':22s} {'min':>13} {'max':>13} {'rango rel':>10}")
    for i, (nm, sg) in enumerate(spec):
        c = allF[ref_idx, i] * sg
        sc = max(abs(c).max(), 1e-12)
        flag = "  <- plano" if (c.max() - c.min()) / sc < 0.05 else ""
        print(f"{nm:22s} {c.min():13.4e} {c.max():13.4e} "
              f"{(c.max()-c.min())/sc:9.1%}{flag}")

    if A and diag:
        allA = np.vstack([A[s] for s in seeds if s in A])
        print(f"\n=== diagnóstico (NO optimizados, sobre todas las políticas) ===")
        for o in diag:
            i = names.index(o)
            nm, sg = PRETTY.get(o, (o, 1))
            c = allA[:, i] * sg
            sc = max(abs(c).max(), 1e-12)
            print(f"{nm:22s} {c.min():13.4e} {c.max():13.4e} "
                  f"{(c.max()-c.min())/sc:9.1%}")

    print(f"\n=== aporte y calidad por semilla ===")
    print(f"{'semilla':>8} {'aporta':>8} {'%frente':>9} {'hiperv.':>10} {'eps-adit.':>10}")
    hvs = {}
    for s in seeds:
        P = ((F[s] - lo) / rng)
        P = P[nondominated(P)]
        hvs[s] = hv_mc(P)
        k = int((owner[ref_idx] == s).sum())
        print(f"{s:8d} {k:8d} {100*k/len(ref):8.1f}% {hvs[s]:10.4f} "
              f"{eps_add(P, ref):10.4f}")

    h = np.array(list(hvs.values()))
    cv = 100 * h.std() / h.mean()
    hv_ref = hv_mc(ref)
    print(f"\nhipervolumen: media={h.mean():.4f}  sd={h.std():.4f}  CV={cv:.1f}%")
    if len(seeds) < 2:
        # con una sola semilla el CV es 0 por construccion y no dice nada
        print("  1 sola semilla: la convergencia NO es evaluable (hacen falta >= 3)")
    else:
        print(f"  {'CONVERGIO' if cv < 10 else 'NO convergio (CV alto: sube --evaluations)'}"
              f"  — {len(seeds)} semillas independientes "
              f"{'coinciden' if cv < 10 else 'divergen'}")
        print(f"frente combinado: {hv_ref:.4f}  "
              f"(+{100*(hv_ref/h.mean()-1):.0f}% sobre una semilla sola)")
        print("  -> reportar la UNION de las semillas, no una sola")

    if a.fig:
        _figure(allF[ref_idx], spec, owner[ref_idx], seeds, a.fig)
        print(f"\nfigura: {a.fig}")


def _figure(P, spec, owner, seeds, out: Path):
    """Coordenadas paralelas con lo DESEABLE siempre arriba.

    Cada objetivo se muestra en sus unidades naturales (J1 y J3 vienen negados
    en el .dat porque NSGA minimiza todo; el signo se revierte para mostrarlos).
    La normalizacion se orienta por objetivo, de modo que el borde superior es
    siempre el mejor valor alcanzado en el frente y el inferior el peor —
    independientemente de si el objetivo se maximiza o se minimiza.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V = np.column_stack([P[:, i] * sg for i, (_, sg) in enumerate(spec)])
    labels = [n for n, _ in spec]
    # sg = -1 -> el objetivo venia negado -> en unidades naturales MAS es mejor
    mas_es_mejor = [sg < 0 for _, sg in spec]
    mn, mx = V.min(0), V.max(0)
    rng = np.where(mx - mn > 0, mx - mn, 1.0)

    Vn = np.empty_like(V)
    mejor, peor = np.empty(len(labels)), np.empty(len(labels))
    for k in range(len(labels)):
        if mas_es_mejor[k]:
            Vn[:, k] = (V[:, k] - mn[k]) / rng[k]      # max arriba
            mejor[k], peor[k] = mx[k], mn[k]
        else:
            Vn[:, k] = (mx[k] - V[:, k]) / rng[k]      # min arriba
            mejor[k], peor[k] = mn[k], mx[k]

    ci = labels.index("Costo J4") if "Costo J4" in labels else 0

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                  gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(hspace=0.5)

    cost = V[:, ci]
    norm = plt.Normalize(cost.min(), cost.max())
    for i in np.argsort(-cost):                        # las caras al fondo
        ax.plot(range(len(labels)), Vn[i], color=plt.cm.viridis_r(norm(cost[i])),
                alpha=0.35, lw=1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([n + ("\n(maximizar)" if b else "\n(minimizar)")
                        for n, b in zip(labels, mas_es_mejor)], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["peor", "MEJOR"], fontsize=9)
    for x in range(len(labels)):
        ax.axvline(x, color="0.8", lw=0.8, zorder=0)
        ax.text(x, 1.05, f"{mejor[x]:,.4g}", ha="center", fontsize=8,
                color="#1a6b1a", fontweight="bold")
        ax.text(x, -0.10, f"{peor[x]:,.4g}", ha="center", fontsize=8, color="#8b1a1a")
    ax.axhline(1.0, color="#1a6b1a", lw=0.8, ls=":", alpha=0.6)
    ax.axhline(0.0, color="#8b1a1a", lw=0.8, ls=":", alpha=0.6)
    ax.set_ylim(-0.16, 1.12)
    ax.set_title(f"Frente robusto combinado — {len(P)} políticas de {len(seeds)} "
                 f"semillas\nel borde SUPERIOR es siempre el mejor valor del frente",
                 pad=26, fontsize=12)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis_r"), ax=ax,
                 label="Costo J4 (oscuro = mas barato)", pad=0.01)

    di = labels.index("Deficit AP") if "Deficit AP" in labels else (
        labels.index("Déficit AP") if "Déficit AP" in labels else min(1, len(labels)-1))
    k = next((j for j in range(len(labels)) if j not in (ci, di)), 0)
    sc = ax2.scatter(V[:, ci], V[:, di], c=V[:, k], cmap="plasma", s=26,
                     edgecolor="k", linewidth=0.25)
    ax2.set_xlabel(labels[ci] + "  (menor es mejor →)")
    ax2.set_ylabel(labels[di] + "  (menor es mejor →)")
    ax2.invert_xaxis(); ax2.invert_yaxis()      # lo mejor hacia arriba-derecha
    ax2.set_title("Trade-off dominante: costo vs deficit  "
                  "(arriba-derecha = mejor en ambos)")
    ax2.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax2, label=labels[k])
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
