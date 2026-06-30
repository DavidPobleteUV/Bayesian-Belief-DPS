# -*- coding: utf-8 -*-
"""
_extract_series_one.py — helper for make_comparison_plots.py. Re-simulates the
selected Pareto policies of ONE variant (under its own checkpoint, set via the
DPS_CKPT env BEFORE importing config) and dumps GW-storage / unmet / agri series
+ objective metadata to an .npz for overlay.

  DPS_CKPT=<ckpt> python weap_dps/_extract_series_one.py --pareto <combined.dat> \
        --select knee --out runs_weap/compare/series_<variant>.npz
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import (
    ZARR_TEMPLATE_PATH, WARMUP_WEEKS, SPIN_UP_YEARS, WEEKS_PER_YEAR,
    ANALYSIS_HORIZON_Y, USD_CLP_RATE,
)
from weap_dps.pipe_simulation_weap import PipeWEAP
from weap_dps.plot_timeseries_from_pareto import (
    evaluate_solution, extract_gw_storage_total, extract_unmet_ap_total,
    extract_agri_production_annual,
)

ap = argparse.ArgumentParser()
ap.add_argument("--pareto", type=Path, required=True)
ap.add_argument("--select", default="knee", choices=["knee", "min_unmet", "min_cost"])
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

sols = pickle.load(open(args.pareto, "rb"))["result"]
O = np.array([o for _, o in sols], dtype=float)   # [J1neg, J2, J3neg, J4, J5]

# pick ONE representative policy
if args.select == "min_unmet":
    sidx = int(np.argmin(O[:, 1]))
elif args.select == "min_cost":
    sidx = int(np.argmin(O[:, 3]))
else:  # knee: min normalized distance to the ideal point on (J2 unmet, J4 cost)
    a = O[:, [1, 3]].astype(float)
    norm = (a - a.min(0)) / (np.ptp(a, 0) + 1e-12)
    sidx = int(np.argmin((norm ** 2).sum(1)))

pipe = PipeWEAP(template_path=ZARR_TEMPLATE_PATH)
tg = pipe.surrogate.target_names_gw; ts = pipe.surrogate.target_names_surf
vars_, objs = sols[sidx]
out = evaluate_solution(pipe, vars_)
gw = extract_gw_storage_total(out["gw_denorm"], tg)
unmet = extract_unmet_ap_total(out["surf_denorm"], ts)
agri = extract_agri_production_annual(out["surf_denorm"], ts)
J1, J2, J3, J4, J5 = objs

np.savez(
    args.out,
    gw=gw if gw is not None else np.zeros(1),
    unmet=unmet if unmet is not None else np.zeros(1),
    agri=agri if agri is not None else np.zeros(1),
    sol_idx=sidx,
    J1_Mm3=-J1 / 1e6, J4_MUSD=J4 / USD_CLP_RATE / 1e6,
    J2=J2, J3_MUSD=-J3 / USD_CLP_RATE / 1e6, J5=J5,
    actions=out["actions_history"],
    decision_start=WARMUP_WEEKS + SPIN_UP_YEARS * WEEKS_PER_YEAR,
)
print(f"  extracted sol#{sidx} ({args.select})  J1={-J1/1e6:.1f}Mm3 J4={J4/USD_CLP_RATE/1e6:.1f}MUSD -> {args.out.name}")
