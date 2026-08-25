# -*- coding: utf-8 -*-
"""
figuras_carteras.py — cinco figuras del frente decodificado.

El gráfico habitual de "% de runs en que gana la acción X" supone que las
acciones COMPITEN. Aquí se complementan: el frente usa cinco carteras de las
dieciséis posibles y el pozo aparece en todas. La pregunta correcta no es qué
acción gana sino cuáles son incondicionales y cuáles contingentes, y estas
figuras están construidas alrededor de eso.

  1. carteras.png    — tipo UpSet: frecuencia por cartera + matriz de membresía
  2. paralelas.png   — frente en coordenadas paralelas, coloreado por cartera
  3. sendas.png      — políticas x años, cuándo entra y sale cada acción
  4. migracion.png   — aluvial: cómo cambia la cartera entre contextos
  5. robustez.png    — criterio de dominio por cartera

Uso:
    python weap_dps/figuras_carteras.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NPZ = Path("results/frente_decodificado.npz")
OUT = Path("results/figuras")
CORTO = {"act_desalacion_costera": "Desal. costera",
         "act_desalacion_completa": "Desal. completa",
         "act_nuevo_pozo_a_5km": "Pozo 5 km",
         "act_acuerdo": "Acuerdo"}
OBJ = {"J2_unmet_ap": "Déficit AP", "J3_agri_value": "Valor agrícola",
       "J4_supply_cost": "Costo", "J51_mean_town_fail": "Semanas falla",
       "J52_worst_year_frac": "Peor año"}
# Menor es mejor salvo J3 (valor agrícola, que se maximiza).
MAXIMIZAR = {"J3_agri_value"}
CTX_NOM = {"megasequia": "Megasequía (−42 %)",
           "sequia_extrema": "Sequía extrema (−72 %)"}


def cargar():
    d = np.load(NPZ, allow_pickle=True)
    acc = [str(x) for x in d["acciones"]]
    ctx = [str(x) for x in d["ctx"]]
    obj = [str(x) for x in d["objetivos"]]
    if len(obj) != d["F"].shape[1]:
        # npz de una version anterior guardo los 7 nombres con F de 5 columnas.
        from weap_dps.config_weap import OBJECTIVES_OPTIMIZED
        obj = list(OBJECTIVES_OPTIMIZED)
    hist = d["hist"]                       # (n_pol, n_ctx, n_años, n_acc)
    activa = hist.max(axis=2) > 0          # (n_pol, n_ctx, n_acc)
    return d["F"], hist, activa, acc, ctx, obj, d["idx_front"]


def etiqueta(fila, acc):
    n = [CORTO[a] for a, v in zip(acc, fila) if v]
    return " + ".join(n) if n else "(ninguna)"


# ── 1. carteras (UpSet) ──────────────────────────────────────────────────────
def fig_carteras(activa, acc, ctx):
    n_ctx = len(ctx)
    fig, axes = plt.subplots(2, n_ctx, figsize=(6.2 * n_ctx, 7),
                             gridspec_kw={"height_ratios": [2.4, 1]}, sharex="col")
    if n_ctx == 1:
        axes = axes.reshape(2, 1)
    for c in range(n_ctx):
        A = activa[:, c, :]
        cart, cuenta = np.unique(A, axis=0, return_counts=True)
        orden = np.argsort(-cuenta)
        cart, cuenta = cart[orden], cuenta[orden]
        pct = 100 * cuenta / len(A)
        x = np.arange(len(cart))

        ax = axes[0, c]
        ax.bar(x, pct, color="#3b6ea5", width=0.62)
        for i, (p, n) in enumerate(zip(pct, cuenta)):
            ax.text(i, p + 0.8, f"{p:.0f}%\n(n={n})", ha="center",
                    va="bottom", fontsize=8.5)
        ax.set_ylim(0, max(pct) * 1.28)
        ax.set_ylabel("% del frente no dominado")
        ax.set_title(CTX_NOM.get(ctx[c], ctx[c]), fontsize=11, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        ax = axes[1, c]
        for j in range(len(acc)):
            ax.axhline(j, color="#eeeeee", lw=6, zorder=0)
        for i, fila in enumerate(cart):
            idx = np.where(fila)[0]
            ax.scatter([i] * len(idx), idx, s=70, color="#3b6ea5", zorder=3)
            if len(idx) > 1:
                ax.plot([i, i], [idx.min(), idx.max()], color="#3b6ea5",
                        lw=2, zorder=2)
            gris = np.where(~fila)[0]
            ax.scatter([i] * len(gris), gris, s=70, color="#cccccc", zorder=3)
        ax.set_yticks(range(len(acc)))
        ax.set_yticklabels([CORTO[a] for a in acc], fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([])
        ax.set_ylim(-0.6, len(acc) - 0.4)
        ax.invert_yaxis()
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.tick_params(left=False)
    fig.suptitle("Carteras de acciones sobre el frente no dominado",
                 fontsize=13, weight="bold")
    # El pie se calcula, no se afirma: el pozo aparece en 358 de 359 bajo
    # megasequía y en las 359 bajo sequía extrema, y decir "en todas" seria
    # falso en el primer contexto.
    frac = [100 * activa[:, c, acc.index("act_nuevo_pozo_a_5km")].mean()
            for c in range(activa.shape[1])]
    fig.text(0.5, 0.005, "Cada columna es una COMBINACIÓN, no una acción suelta. "
             f"El pozo está en el {min(frac):.1f}–{max(frac):.1f} % de las "
             "políticas: es la medida sin arrepentimiento.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT / "1_carteras.png", dpi=160); plt.close(fig)


# ── 2. coordenadas paralelas coloreadas por cartera ──────────────────────────
def fig_paralelas(F, activa, acc, obj, ctx_i=0):
    A = activa[:, ctx_i, :]
    cart, inv, cuenta = np.unique(A, axis=0, return_inverse=True, return_counts=True)
    orden = np.argsort(-cuenta)
    rank = {int(o): r for r, o in enumerate(orden)}
    colores = plt.cm.tab10(np.linspace(0, 1, 10))

    # Normaliza a [0,1] con "mejor arriba" en todos los ejes.
    Z = np.zeros_like(F, dtype=float)
    for k, o in enumerate(obj):
        v = F[:, k]
        rng = v.max() - v.min()
        z = (v - v.min()) / (rng if rng > 1e-12 else 1.0)
        # El .dat guarda todo en convención de minimizar: invertir para que
        # arriba sea siempre mejor, y anotar el sentido en la etiqueta.
        Z[:, k] = 1 - z
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(obj))
    for i in range(len(F)):
        r = rank[int(inv[i])]
        ax.plot(x, Z[i], color=colores[r % 10], alpha=0.28, lw=1.0, zorder=2)
    for r, o in enumerate(orden):
        ax.plot([], [], color=colores[r % 10], lw=2.4,
                label=f"{etiqueta(cart[o], acc)}  (n={cuenta[o]})")
    ax.set_xticks(x)
    ax.set_xticklabels([OBJ.get(o, o) for o in obj], fontsize=10)
    ax.set_ylabel("mejor  ←→  peor   (normalizado por rango)")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["peor", "mejor"])
    ax.set_ylim(-0.05, 1.05)
    for xi in x:
        ax.axvline(xi, color="#dddddd", lw=1, zorder=1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9,
              frameon=False)
    ax.set_title("Frente de Pareto por cartera de acciones — "
                 f"{CTX_NOM.get(list(CTX_NOM)[ctx_i], '')}",
                 fontsize=12, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "2_1_paralelas.png", dpi=160, bbox_inches="tight"); plt.close(fig)


# ── 2.2 pequeños múltiplos: un panel por cartera ─────────────────────────────
def fig_paralelas_facetas(F, activa, acc, obj, ctx_i=0, n_min=10):
    """Misma información que 2.1 pero sin superposición.

    Con 359 líneas y ocho carteras, el gráfico único satura y las carteras
    minoritarias quedan tapadas por las dominantes. Aquí cada panel aísla una
    cartera contra el frente completo en gris, de modo que su firma —qué ejes
    mejora y cuáles sacrifica— se lee de inmediato.
    """
    A = activa[:, ctx_i, :]
    cart, inv, cuenta = np.unique(A, axis=0, return_inverse=True, return_counts=True)
    orden = np.argsort(-cuenta)
    grandes = [int(o) for o in orden if cuenta[o] >= n_min]
    resto = [int(o) for o in orden if cuenta[o] < n_min]
    paneles = [(o, etiqueta(cart[o], acc), cuenta[o]) for o in grandes]
    if resto:
        paneles.append((None, "otras carteras", sum(cuenta[o] for o in resto)))

    Z = np.zeros_like(F, dtype=float)
    for k in range(F.shape[1]):
        v = F[:, k]; rng = v.max() - v.min()
        Z[:, k] = 1 - (v - v.min()) / (rng if rng > 1e-12 else 1.0)

    n = len(paneles)
    ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.3 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(F.shape[1])
    colores = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, (o, nom, n_pol) in enumerate(paneles):
        ax = axes[k]
        for i in range(len(F)):                      # frente completo, de fondo
            ax.plot(x, Z[i], color="#dddddd", lw=0.6, alpha=0.55, zorder=1)
        mask = (inv == o) if o is not None else np.isin(inv, resto)
        for i in np.where(mask)[0]:
            ax.plot(x, Z[i], color=colores[k % 10], lw=1.1, alpha=0.75, zorder=3)
        med = np.median(Z[mask], axis=0)
        ax.plot(x, med, color="black", lw=2.4, zorder=4)
        ax.set_title(f"{nom}\n(n={n_pol}, {100*n_pol/len(F):.0f} %)", fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels([OBJ.get(o_, o_) for o_ in obj], fontsize=8, rotation=30,
                           ha="right")
        ax.set_ylim(-0.05, 1.05); ax.set_yticks([0, 1])
        ax.set_yticklabels(["peor", "mejor"], fontsize=8)
        for xi in x:
            ax.axvline(xi, color="#eeeeee", lw=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Firma de cada cartera sobre el frente — "
                 f"{CTX_NOM.get(list(CTX_NOM)[ctx_i], '')}",
                 fontsize=13, weight="bold")
    fig.text(0.5, 0.005, "Gris: las 359 políticas. Color: la cartera del panel. "
             "Negro: su mediana. Todos los ejes con MEJOR arriba.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT / "2_2_paralelas_facetas.png", dpi=160); plt.close(fig)


# ── 3. sendas de adaptación ──────────────────────────────────────────────────
def fig_sendas(hist, acc, ctx_i=0, n_muestra=60):
    H = hist[:, ctx_i, :, :]                     # (n_pol, años, acc)
    orden = np.lexsort([H[:, :, j].sum(1) for j in range(len(acc))][::-1])
    sel = orden[np.linspace(0, len(orden) - 1, min(n_muestra, len(orden))).astype(int)]
    fig, axes = plt.subplots(1, len(acc), figsize=(4.0 * len(acc), 5.4), sharey=True)
    cols = ["#2c7fb8", "#d95f0e", "#31a354", "#756bb1"]
    for j, a in enumerate(acc):
        ax = axes[j]
        ax.imshow(H[sel, :, j], aspect="auto", cmap=
                  matplotlib.colors.ListedColormap(["#f2f2f2", cols[j]]),
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(CORTO[a], fontsize=11, color=cols[j], weight="bold")
        ax.set_xlabel("año de decisión")
        ax.set_xticks([0, 10, 20, 32])
        ax.set_xticklabels([2027, 2037, 2047, 2059], fontsize=9)
        if j == 0:
            ax.set_ylabel(f"políticas del frente (muestra de {len(sel)})")
        ax.set_yticks([])
    fig.suptitle("Sendas de adaptación: cuándo opera cada acción",
                 fontsize=13, weight="bold")
    fig.text(0.5, 0.01, "Cada fila es una política. El color marca los años en que "
             "la acción está operando: las bandas interrumpidas son apagados y "
             "reencendidos, imposibles bajo un calendario fijo.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUT / "3_sendas.png", dpi=160); plt.close(fig)


# ── 4. migración de carteras entre contextos ─────────────────────────────────
def fig_migracion(activa, acc, ctx):
    if activa.shape[1] < 2:
        return
    A0, A1 = activa[:, 0, :], activa[:, 1, :]
    cart = np.unique(np.vstack([A0, A1]), axis=0)
    idx = {tuple(c): i for i, c in enumerate(cart)}
    n = len(cart)
    c0 = np.array([idx[tuple(r)] for r in A0])
    c1 = np.array([idx[tuple(r)] for r in A1])
    tot0 = np.array([(c0 == i).sum() for i in range(n)])
    tot1 = np.array([(c1 == i).sum() for i in range(n)])
    orden = np.argsort(-(tot0 + tot1))
    pos = {int(o): k for k, o in enumerate(orden)}
    colores = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(10, 6))
    alto = 0.72
    for lado, tot in ((0, tot0), (1, tot1)):
        y = 0.0
        for o in orden:
            h = tot[o] / len(c0) * 100
            ax.add_patch(plt.Rectangle((lado * 6 - 0.45, y), 0.9, h * alto,
                                       color=colores[pos[int(o)] % 10], alpha=0.9))
            if h > 1:
                ax.text(lado * 6, y + h * alto / 2, f"{h:.0f}%", ha="center",
                        va="center", fontsize=9, color="white", weight="bold")
            y += h * alto + 1.2
    # flujos
    y0 = {}; acum0 = 0.0
    for o in orden:
        y0[int(o)] = acum0; acum0 += tot0[o] / len(c0) * 100 * alto + 1.2
    y1 = {}; acum1 = 0.0
    for o in orden:
        y1[int(o)] = acum1; acum1 += tot1[o] / len(c0) * 100 * alto + 1.2
    off0 = dict(y0); off1 = dict(y1)
    for a in orden:
        for b in orden:
            m = int(((c0 == a) & (c1 == b)).sum())
            if not m:
                continue
            h = m / len(c0) * 100 * alto
            ax.fill_betweenx([0, 1], 0, 0)   # no-op para mantener límites
            ys, ye = off0[int(a)], off1[int(b)]
            t = np.linspace(0, 1, 60)
            sup = ys + (ye - ys) * (3 * t**2 - 2 * t**3)
            ax.fill_between(0.45 + t * 5.1, sup, sup + h,
                            color=colores[pos[int(a)] % 10], alpha=0.30, lw=0)
            off0[int(a)] += h; off1[int(b)] += h
    for lado, nom in ((0, CTX_NOM.get(ctx[0], ctx[0])),
                      (1, CTX_NOM.get(ctx[1], ctx[1]))):
        ax.text(lado * 6, -3.5, nom, ha="center", fontsize=11, weight="bold")
    ax.legend(handles=[Patch(color=colores[pos[int(o)] % 10],
                             label=etiqueta(cart[o], acc)) for o in orden],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              fontsize=9, frameon=False)
    ax.set_xlim(-1.2, 7.2); ax.set_ylim(-6, max(acum0, acum1) + 2)
    ax.axis("off")
    ax.set_title("Migración de carteras al endurecerse la sequía",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "4_migracion.png", dpi=160, bbox_inches="tight"); plt.close(fig)


# ── 5. robustez por cartera ──────────────────────────────────────────────────
def fig_robustez(activa, acc, idx_front):
    rb = Path("results/robustez_iter1_fix2050/robustez.npz")
    if not rb.exists():
        print("  (sin robustez.npz — se omite la figura 5)")
        return
    d = np.load(rb, allow_pickle=True)
    sel, dom = d["sel"], d["dominio"] * 100
    pos = {int(v): k for k, v in enumerate(idx_front)}
    pares = [(pos[int(s)], dm) for s, dm in zip(sel, dom) if int(s) in pos]
    if not pares:
        print("  (los índices de robustez no mapean al frente — se omite la 5)")
        return
    ii = np.array([p for p, _ in pares]); dd = np.array([v for _, v in pares])
    A = activa[ii, 0, :]
    cart, inv, cuenta = np.unique(A, axis=0, return_inverse=True, return_counts=True)
    orden = np.argsort(-cuenta)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for k, o in enumerate(orden):
        v = dd[inv == o]
        ax.scatter(np.full(len(v), k) + np.random.uniform(-.09, .09, len(v)),
                   v, s=48, alpha=.75, color=plt.cm.tab10(k % 10), zorder=3)
        ax.hlines(np.median(v), k - .25, k + .25, color="black", lw=2.2, zorder=4)
    ax.axhline(90, color="#c0392b", ls="--", lw=1.2)
    ax.text(len(orden) - .5, 90.8, "umbral 90 %", color="#c0392b", fontsize=9,
            ha="right")
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([etiqueta(cart[o], acc).replace(" + ", "\n+ ")
                        for o in orden], fontsize=8.5)
    ax.set_ylabel("criterio de dominio (% de los 81 futuros)")
    ax.set_title("Robustez por cartera — barra negra = mediana",
                 fontsize=12, weight="bold")
    # robustness_test elige sus 40 politicas del conjunto COMPLETO (600), que
    # incluye dominadas; solo las no dominadas tienen cartera decodificada.
    ax.text(0.5, -0.30, f"{len(dd)} de las {len(sel)} políticas evaluadas en "
            "robustez pertenecen al frente no dominado y tienen cartera asignada.",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "5_robustez.png", dpi=160); plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    F, hist, activa, acc, ctx, obj, idx_front = cargar()
    print(f"frente decodificado: {len(F)} políticas, {len(ctx)} contextos")
    for c, nom in enumerate(ctx):
        A = activa[:, c, :]
        cart, cuenta = np.unique(A, axis=0, return_counts=True)
        print(f"  {nom}: {len(cart)} carteras distintas de {2**len(acc)}")
        for o in np.argsort(-cuenta):
            print(f"      {100*cuenta[o]/len(A):5.1f}%  {etiqueta(cart[o], acc)}")
    fig_carteras(activa, acc, ctx);      print("  1_carteras.png")
    fig_paralelas(F, activa, acc, obj);          print("  2_1_paralelas.png")
    fig_paralelas_facetas(F, activa, acc, obj);  print("  2_2_paralelas_facetas.png")
    fig_sendas(hist, acc);               print("  3_sendas.png")
    fig_migracion(activa, acc, ctx);     print("  4_migracion.png")
    fig_robustez(activa, acc, idx_front); print("  5_robustez.png")
    print(f"\nfiguras en {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
