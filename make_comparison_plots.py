# -*- coding: utf-8 -*-
"""
make_comparison_plots.py — compare the 4 DPS variants (v2, v2.3, v3, v3.3):
combines per-seed Pareto fronts into one non-dominated front per variant, then
produces (1) Pareto overlays across variants and (2) GW-storage / unmet / agri
timeseries overlays for one representative policy per variant.

Timeseries re-simulation runs per-variant in a subprocess with the right
DPS_CKPT (GW/unmet/agri depend on the base checkpoint, not the waterfall).

  python make_comparison_plots.py --src runs_weap/prod --out runs_weap/compare
  python make_comparison_plots.py --src runs_weap/smoke --out runs_weap/compare_smoke --smoke
"""
from __future__ import annotations
import argparse, glob, os, pickle, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = ROOT / "venv_DPS" / "Scripts" / "python.exe"
V2_CKPT = "../WEAP_HydroMLP_RecursiveGW/runs/iter07_v2_clean/best_model-epoch=021-val_loss=0.0623.ckpt"
V3_CKPT = "../WEAP_HydroMLP_RecursiveGW/runs/iter07_v3_clean/best_model-epoch=011-val_loss=0.0638.ckpt"
VARIANTS = {  # name -> (ckpt, color)
    "v2":   (V2_CKPT, "#1f77b4"), "v2_3": (V2_CKPT, "#7fbfff"),
    "v3":   (V3_CKPT, "#d62728"), "v3_3": (V3_CKPT, "#ff9896"),
}
OBJ = ["J1 storage[Mm³]", "J2 unmet", "J3 agri", "J4 cost[bnCLP]", "J5 failwk"]


def nondominated(O):
    keep = np.ones(len(O), bool)
    for i in range(len(O)):
        if not keep[i]:
            continue
        dom = np.all(O <= O[i], 1) & np.any(O < O[i], 1)
        dom[i] = False
        keep[dom] = False
    return keep


def combine_variant(src: Path, name: str):
    files = sorted(glob.glob(str(src / f"pareto_{name}_seed*.dat"))) or \
            sorted(glob.glob(str(src / f"pareto_{name}.dat")))   # smoke: single file
    if not files:
        return None
    allsol = []
    for f in files:
        allsol += pickle.load(open(f, "rb"))["result"]
    O = np.array([o for _, o in allsol], float)
    keep = nondominated(O)
    front = [allsol[i] for i in np.where(keep)[0]]
    pickle.dump({"result": front}, open(src / f"combined_{name}.dat", "wb"))
    return np.array([o for _, o in front], float)


def disp(O):
    """to display units: J1->+Mm3, J4->bn CLP."""
    D = O.copy(); D[:, 0] = -O[:, 0] / 1e6; D[:, 3] = O[:, 3] / 1e9; D[:, 1] = O[:, 1] / 1e6
    return D


def plot_pareto_overlays(fronts, out: Path):
    D = {n: disp(O) for n, O in fronts.items()}
    pairs = [(0, 3), (1, 3), (0, 1), (2, 3), (1, 4)]   # (storage,cost)(unmet,cost)(stor,unmet)(agri,cost)(unmet,fail)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (a, b) in zip(axes.flat, pairs):
        for n, M in D.items():
            ax.scatter(M[:, b], M[:, a], s=16, c=VARIANTS[n][1], label=n.replace("_", "."),
                       alpha=.7, edgecolors="none")
        ax.set_xlabel(OBJ[b]); ax.set_ylabel(OBJ[a]); ax.grid(alpha=.25)
    axes.flat[0].legend(title="variant", fontsize=9)
    axes.flat[-1].axis("off")
    fig.suptitle("DPS Pareto fronts — 4 variants (combined seeds, non-dominated)", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "pareto_overlays_4variant.png", dpi=120); plt.close(fig)
    print("saved pareto_overlays_4variant.png")


def plot_timeseries(series, out: Path):
    import pandas as pd
    from weap_dps.plot_timeseries_from_pareto import _build_dates  # real WEAP calendar (time[])
    dates = lambda n: _build_dates(n)
    specs = [("gw", "GW storage total [m³]", "GW storage trajectory"),
             ("unmet", "Unmet AP [l/s]", "Unmet AP trajectory"),
             ("agri", "Agri production [ton/yr]", "Agri production trajectory")]
    for key, ylab, title in specs:
        fig, ax = plt.subplots(figsize=(13, 5))
        ds = None
        for n, S in series.items():
            arr = S[key]
            if arr is None or arr.size <= 1:
                continue
            ds = dates(len(arr))
            ax.plot(ds, arr, color=VARIANTS[n][1], lw=1.4, alpha=.85,
                    label=f"{n.replace('_','.')}  (J1={S['J1_Mm3']:.1f}Mm³ J4={S['J4_MUSD']:.0f}M)")
        if ds is not None and 0 < int(S["decision_start"]) < len(ds):
            ax.axvline(ds[int(S["decision_start"])], color="gray", ls="--", alpha=.6, lw=1)
        ax.set_xlabel("Date"); ax.set_ylabel(ylab); ax.set_title(title, fontweight="bold")
        ax.grid(alpha=.3); ax.legend(fontsize=8, title="variant (knee policy)")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        fig.tight_layout(); fig.savefig(out / f"ts_{key}_4variant.png", dpi=120); plt.close(fig)
        print(f"saved ts_{key}_4variant.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--select", default="knee")
    ap.add_argument("--smoke", action="store_true", help="rescale J4 by per-variant cal (smoke used 1.22)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    CAL = {"v2": 1.149, "v2_3": 1.189, "v3": 1.032, "v3_3": 1.184}
    fronts = {}
    for n in VARIANTS:
        O = combine_variant(args.src, n)
        if O is None:
            print(f"  (no files for {n}, skipping)"); continue
        if args.smoke:
            O = O.copy(); O[:, 3] = O[:, 3] / 1.22 * CAL[n]   # smoke baked-in 1.22 -> corrected
        fronts[n] = O
        print(f"  {n}: combined front = {len(O)} non-dominated")
    if not fronts:
        print("no fronts found"); return

    plot_pareto_overlays(fronts, args.out)

    # timeseries: re-simulate the representative policy per variant (subprocess w/ right ckpt)
    series = {}
    for n in fronts:
        comb = args.src / f"combined_{n}.dat"
        npz = args.out / f"series_{n}.npz"
        env = dict(os.environ, DPS_CKPT=VARIANTS[n][0])
        r = subprocess.run([str(PY), "weap_dps/_extract_series_one.py", "--pareto", str(comb),
                            "--select", args.select, "--out", str(npz)],
                           cwd=str(ROOT), env=env, capture_output=True, text=True)
        if npz.exists():
            series[n] = dict(np.load(npz, allow_pickle=True))
        else:
            print(f"  ts extract FAILED {n}:\n{r.stderr[-500:]}")
    if series:
        plot_timeseries(series, args.out)

    # summary table
    rows = []
    for n, O in fronts.items():
        D = disp(O)
        rows.append({"variant": n.replace("_", "."), "front_n": len(O),
                     "J1_med_Mm3": np.median(D[:, 0]), "J4_med_bnCLP": np.median(D[:, 3]),
                     "J4_min": D[:, 3].min(), "J2_med": np.median(D[:, 1]), "J5_med": np.median(O[:, 4])})
    pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False)
    print("saved summary.csv\n", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
