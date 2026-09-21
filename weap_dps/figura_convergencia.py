# -*- coding: utf-8 -*-
"""
figura_convergencia.py — Hipervolumen contra evaluaciones.

Grafica la trayectoria HV(nfe) que registran las corridas de eps-NSGA-II, que es
la evidencia de si el presupuesto alcanzo. Lee tanto los .dat finales como los
.ckpt de corridas EN CURSO, asi que sirve para mirar el avance sin detener nada.

Por que esta figura y no la dispersion del HV entre semillas: esa ultima acota
la varianza entre semillas, no el sesgo comun a todas. Cinco semillas pueden
converger consistentemente a la misma region subóptima y dar un CV excelente.
Lo que distingue "convergio" de "se quedo sin presupuesto" es la PENDIENTE al
final de la curva.

La linea horizontal es el HV de la corrida NSGA-II de iter02 sobre la misma caja
fija, para leer de un vistazo si eps-NSGA-II la supera y a partir de cuantas
evaluaciones.

Uso:
    python weap_dps/figura_convergencia.py
    python weap_dps/figura_convergencia.py --eps "runs_weap/eps_iter02/*.ckpt"
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weap_dps.config_weap import HV_MAXIMUM, HV_MINIMUM  # noqa: E402
from weap_dps.hv_utils import hipervolumen_mc  # noqa: E402

OUT = Path("results/figuras")
COLORES = ["#2874a6", "#c0392b", "#27ae60", "#8e44ad", "#d68910",
           "#16a085", "#7f8c8d"]


def historias(patron: str) -> list[tuple[str, list[dict]]]:
    out = []
    for f in sorted(glob.glob(patron)):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        h = d.get("hv_history") or []
        if h:
            out.append((Path(f).stem, h))
        else:
            print(f"  (sin historia de HV: {Path(f).name})")
    return out


def referencia_nsga(patron: str) -> float | None:
    """HV de la union de los frentes NSGA-II, sobre la misma caja fija."""
    A = []
    for f in sorted(glob.glob(patron)):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        A.extend(o for _, o in d.get("result", []))
    if not A:
        return None
    hv, _ = hipervolumen_mc(np.array(A, float), HV_MINIMUM, HV_MAXIMUM)
    return hv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="runs_weap/eps_iter02/*",
                   help="patron de .dat o .ckpt con historia de HV")
    ap.add_argument("--nsga", default="results/iter02_pareto_seed*.dat")
    ap.add_argument("--out", type=Path, default=OUT / "C1_convergencia_hv.png")
    a = ap.parse_args()

    H = historias(a.eps)
    if not H:
        print(f"Sin historias de HV en {a.eps}")
        return 1
    ref = referencia_nsga(a.nsga)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                                  gridspec_kw={"width_ratios": [1.45, 1]})

    for k, (nombre, h) in enumerate(H):
        c = COLORES[k % len(COLORES)]
        nfe = [r["nfe"] for r in h]
        ax.plot(nfe, [r["hv"] for r in h], lw=1.7, color=c, label=nombre)
        ax2.plot(nfe, [r["archivo"] for r in h], lw=1.7, color=c)

    if ref is not None:
        ax.axhline(ref, color="#1a1a1a", ls="--", lw=1.4)
        ax.text(0.015, ref, f" NSGA-II, union de 5 semillas a 4.000 ev.: {ref:.4f}",
                transform=ax.get_yaxis_transform(), va="bottom", fontsize=8.5,
                color="#1a1a1a")

    ax.set_xlabel("evaluaciones", fontsize=9)
    ax.set_ylabel("hipervolumen (fraccion de la caja)", fontsize=9)
    ax.set_title("Convergencia: si la curva sigue subiendo, falto presupuesto",
                 fontsize=10, weight="bold")
    ax.legend(fontsize=8, frameon=False, ncol=2)

    ax2.set_xlabel("evaluaciones", fontsize=9)
    ax2.set_ylabel("soluciones en el archivo eps", fontsize=9)
    ax2.set_title("Tamano del archivo", fontsize=10, weight="bold")

    for e in (ax, ax2):
        e.tick_params(labelsize=8)
        e.spines[["top", "right"]].set_visible(False)
        e.grid(alpha=.25, lw=.6)

    fig.suptitle("Robust DPS con eps-NSGA-II — hipervolumen sobre caja fija "
                 "(estimado por Monte Carlo)", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    plt.close(fig)
    print(f"figura: {a.out}")

    # Pendiente al final: el numero que decide si vale la pena seguir.
    print("\nganancia de HV en el ultimo cuarto del presupuesto recorrido:")
    for nombre, h in H:
        v = [r["hv"] for r in h]
        k = max(1, len(v) // 4)
        ini, fin = v[-k - 1], v[-1]
        g = 100 * (fin - ini) / ini if ini else float("nan")
        print(f"  {nombre:<26} nfe {h[-1]['nfe']:>6}  HV {fin:.5f}  {g:+.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
