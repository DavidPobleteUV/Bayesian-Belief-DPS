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

# Nombre y signo de cada objetivo. J1 y J3 se optimizan negados (NSGA minimiza
# todo), así que para MOSTRARLOS hay que devolverles el signo.
OBJ_7 = [("GW storage", -1), ("Déficit AP", 1), ("Valor agrícola", -1),
         ("Costo J4", 1), ("Semanas falla/pueblo", 1), ("Peor año", 1),
         ("Salinidad", 1)]
OBJ_6 = [("GW storage", -1), ("Déficit AP", 1), ("Valor agrícola", -1),
         ("Costo J4", 1), ("Semanas falla", 1), ("Salinidad", 1)]


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
    seeds, F, V, el = [], {}, {}, {}
    for f in sorted(d.glob("pareto_seed*.dat")):
        s = int(f.stem.replace("pareto_seed", ""))
        obj = pickle.load(open(f, "rb"))
        F[s] = np.array([r[1] for r in obj["result"]], dtype=float)
        V[s] = np.array([r[0] for r in obj["result"]], dtype=float)
        el[s] = obj.get("elapsed", float("nan")) / 3600.0
        seeds.append(s)
    if not seeds:
        raise SystemExit(f"No hay pareto_seed*.dat en {d}")
    return seeds, F, V, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--fig", type=Path, default=None, help="PNG del frente")
    a = ap.parse_args()

    seeds, F, V, el = load(a.dir)
    n_obj = F[seeds[0]].shape[1]
    spec = OBJ_7 if n_obj == 7 else OBJ_6
    if len(spec) != n_obj:
        raise SystemExit(f"{n_obj} objetivos: no calza con 6 ni 7")

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
    print(f"  {'CONVERGIO' if cv < 10 else 'NO convergio (CV alto: sube --evaluations)'}"
          f"  — semillas independientes {'coinciden' if cv < 10 else 'divergen'}")
    print(f"frente combinado: {hv_ref:.4f}  "
          f"(+{100*(hv_ref/h.mean()-1):.0f}% sobre una semilla sola)")
    print("  -> reportar la UNION de las semillas, no una sola")

    if a.fig:
        _figure(allF[ref_idx], spec, owner[ref_idx], seeds, a.fig)
        print(f"\nfigura: {a.fig}")


def _figure(P, spec, owner, seeds, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V = np.column_stack([P[:, i] * sg for i, (_, sg) in enumerate(spec)])
    labels = [n for n, _ in spec]
    mn, mx = V.min(0), V.max(0)
    Vn = (V - mn) / np.where(mx - mn > 0, mx - mn, 1)
    ci = [n for n, _ in spec].index("Costo J4")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                  gridspec_kw={"height_ratios": [1.2, 1]})
    fig.subplots_adjust(hspace=0.45)
    cost = V[:, ci]
    norm = plt.Normalize(cost.min(), cost.max())
    for i in np.argsort(-cost):
        ax.plot(range(len(labels)), Vn[i], color=plt.cm.viridis(norm(cost[i])),
                alpha=0.35, lw=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("normalizado (0 = mejor del frente)")
    for x in range(len(labels)):
        ax.axvline(x, color="0.8", lw=0.8, zorder=0)
        ax.text(x, 1.06, f"{mn[x]:,.4g}", ha="center", fontsize=7, color="0.35")
        ax.text(x, -0.12, f"{mx[x]:,.4g}", ha="center", fontsize=7, color="0.35")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Frente robusto combinado — {len(P)} políticas, "
                 f"{len(seeds)} semillas", pad=24)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                 label="Costo J4", pad=0.01)

    di = [n for n, _ in spec].index("Déficit AP")
    sc = ax2.scatter(V[:, ci], V[:, di], c=V[:, 2], cmap="plasma", s=26,
                     edgecolor="k", linewidth=0.25)
    ax2.set_xlabel(labels[ci]); ax2.set_ylabel(labels[di])
    ax2.set_title("Trade-off dominante: costo vs déficit")
    ax2.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax2, label=labels[2])
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
