# -*- coding: utf-8 -*-
"""
robust_pareto_to_rerun.py — exporta propuestas del frente Robust DPS al formato NUEVO
de la re-corrida WEAP (run_XXXX.csv valor-de-uso + master), para run_rerun_one.py.

Selección = 7 representativas (5 extremos + 2 balanceadas) + N políticas que APAGAN
una desaladora o el pozo (comportamiento intermitente). Cada propuesta × 8 climas.

El schedule se escribe DIRECTO del actions_history año a año (respeta cualquier ventana
on/off real, no se aplana). prorrateo_cuenca → prorrateo_shac (comparten branch).

  DPS_CKPT=<v3.ckpt> python robust_pareto_to_rerun.py --pareto runs_weap/robust/pareto_v3_seed42.dat
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from weap_dps.config_weap import ZARR_TEMPLATE_PATH, DECISION_YEARS, SPIN_UP_YEARS
from weap_dps.pipe_simulation_weap import PipeWEAP
from weap_dps.pareto_to_runids import load_pareto, select_representative_solutions

W2Z = Path(__file__).resolve().parent.parent / "WEAP_2_ZARR"
POLDIR = W2Z / "data" / "policies_rerun"
MASTER = W2Z / "data" / "RunIDs_Q_dps_proposals.csv"

ACTS4 = ["desalacion_costera", "desalacion_completa", "prorrateo_shac", "nuevo_pozo_a_5km"]
VALUES = {"desalacion_costera": (0.1, 0.0), "desalacion_completa": (0.3, 0.0),
          "prorrateo_shac": (0.85, 1.0), "nuevo_pozo_a_5km": (120.0, 0.0001)}
HY0, HY1, FIN, START = 2014, 2050, 9999, 2027
INFRA_IDX = {0: "desal_costera", 1: "desal_completa", 4: "nuevo_pozo"}   # cols DPS que apagan
CLIMAS = [("NESM3", "ssp585"), ("EC-Earth3-Veg", "ssp585"), ("ACCESS-CM2", "ssp585"),
          ("MPI-ESM1-2-LR", "ssp585"), ("AWI-CM-1-1-MR", "ssp585"),
          ("NESM3", "ssp245"), ("ACCESS-CM2", "ssp245"), ("MPI-ESM1-2-LR", "ssp245")]
DEM_AGRO, DEM_POB = "Sin cambio en Areas Regadas", "Crecimiento anual regular: 2%"
# sequía prolongada para estresar las propuestas (precip ~10% durante casi todo el horizonte)
DROUGHT_LONG = (0.90, 30, 2027)     # (severity, duration, start_year), mode=extreme
DROUGHT_CLIMAS = [("ACCESS-CM2", "ssp585"), ("MPI-ESM1-2-LR", "ssp585"), ("NESM3", "ssp585")]


def make_row(rid, idx, role, gcm, ssp, drought, block, AH):
    col = cols4_from_history(AH[idx])
    pf = write_schedule(rid, col)
    d = {"ID": rid, "GCM": gcm, "SSP": ssp, "Demanda_Agro": DEM_AGRO, "Demanda_Poblacion": DEM_POB}
    for a in ACTS4:
        act, on, off = on_off(col[a]); d[f"act_{a}"] = act; d[f"on_{a}"] = on; d[f"off_{a}"] = off
    sev, dur, sy = drought if drought else ("", "", "")
    d["drought_severity"] = sev; d["drought_duration"] = dur; d["drought_start_year"] = sy
    d["drought_severity_mode"] = "extreme" if drought else ""
    d["block"] = block; d["policy_file"] = pf; d["pareto_role"] = role
    on_desc = "+".join(f"{a.split('_')[1][:5]}[{on_off(col[a])[1]}-{on_off(col[a])[2]}]"
                       for a in ACTS4 if on_off(col[a])[0]) or "sin_accion"
    dr = f" | SEQUIA sev={sev} dur={dur} start={sy}" if drought else ""
    d["description"] = f"DPS {role} | {on_desc} | clima:{gcm} {ssp}{dr}"
    return d


def is_only_prorr(AH, idx):
    c = cols4_from_history(AH[idx])
    return c["prorrateo_shac"].any() and not any(
        c[a].any() for a in ["desalacion_costera", "desalacion_completa", "nuevo_pozo_a_5km"])


def cols4_from_history(ah: np.ndarray) -> dict:
    """ah (n_years,5) binario → {accion4: array binario por año}. cuenca∪shac."""
    nb = ah[:, :5] > 0.5
    return {"desalacion_costera": nb[:, 0], "desalacion_completa": nb[:, 1],
            "prorrateo_shac": nb[:, 2] | nb[:, 3], "nuevo_pozo_a_5km": nb[:, 4]}


def on_off(seq) -> tuple:
    """(act, on_year, off_year/'fin') del primer ciclo de un array binario por año."""
    idx = np.where(seq)[0]
    if not len(idx):
        return 0, "", ""
    on = START + int(idx[0])
    offs = np.where(np.diff(seq.astype(int)) == -1)[0]
    off = (START + int(offs[0]) + 1) if len(offs) else "fin"
    return 1, on, off


def write_schedule(rid: int, col: dict) -> str:
    fname = f"run_{rid:04d}.csv"
    lines = ["$Columns = Date," + ",".join(ACTS4)]
    for Y in range(HY0, HY1):
        vals = []
        for a in ACTS4:
            on_v, off_v = VALUES[a]; y = Y - START; seq = col[a]
            on = (0 <= y < len(seq)) and bool(seq[y])
            vals.append(repr(on_v if on else off_v))
        for date in (f"4/2/{Y}", f"4/1/{Y + 1}"):
            lines.append(date + "," + ",".join(vals))
    (POLDIR / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", required=True, type=Path)
    ap.add_argument("--start_id", type=int, default=2000)
    ap.add_argument("--n_offswitch", type=int, default=6, help="políticas con apagado a incluir")
    args = ap.parse_args()
    POLDIR.mkdir(parents=True, exist_ok=True)

    pareto, _ = load_pareto(args.pareto)
    pipe = PipeWEAP(template_path=ZARR_TEMPLATE_PATH)

    # re-simular las 100 una vez (rollout determinista)
    AH = []
    for s in pareto:
        pf = pipe._build_policy_from_params(np.asarray(s.variables, float))
        r = pipe.surrogate.rollout_with_policy(X_template=pipe.X_template, policy_fn=pf,
                                               n_years=DECISION_YEARS, action_col_idx=pipe.action_col_idx,
                                               spin_up_years=SPIN_UP_YEARS)
        AH.append(np.asarray(r["actions_history"]))

    # 7 representativas (marca s.role in-place)
    select_representative_solutions(pareto, n_balanced=2)
    chosen = [(i, s.role) for i, s in enumerate(pareto) if getattr(s, "role", "")]

    # políticas que APAGAN desal/pozo — hasta 2 por acción de infra, offyears distintos
    repr_idx = {i for i, _ in chosen}
    off_pool = {}
    for i, ah in enumerate(AH):
        if i in repr_idx:
            continue
        nb = ah[:, :5] > 0.5
        for c, name in INFRA_IDX.items():
            seq = nb[:, c].astype(int)
            if (np.diff(seq) == -1).any():
                _, on, off = on_off(seq)
                off_pool.setdefault(name, []).append((i, off))
    off_sel = []
    for name, lst in off_pool.items():
        lst = sorted(set(lst), key=lambda t: t[1])
        for i, off in [lst[0], lst[-1]][:2]:        # el de apagado más temprano y más tardío
            if (i, name) not in [(x[0], x[1]) for x in off_sel]:
                off_sel.append((i, name, off))
    off_sel = off_sel[:args.n_offswitch]
    for i, name, off in off_sel:
        chosen.append((i, f"apaga_{name}@{off}_#{i}"))

    # sacar propuestas solo-prorrateo (p. ej. balanced_2)
    chosen = [(i, r) for i, r in chosen if not is_only_prorr(AH, i)]
    print(f"propuestas (sin solo-prorrateo): {len(chosen)}  + sequía prolongada")

    rows = []; rid = args.start_id
    for idx, role in chosen:                        # bloque 1: clima GCM puro × 8 climas
        for gcm, ssp in CLIMAS:
            rows.append(make_row(rid, idx, role, gcm, ssp, None, "dps_proposal", AH)); rid += 1
    for idx, role in chosen:                        # bloque 2: sequía prolongada × 3 climas ssp585
        for gcm, ssp in DROUGHT_CLIMAS:
            rows.append(make_row(rid, idx, role, gcm, ssp, DROUGHT_LONG, "dps_drought", AH)); rid += 1

    df = pd.DataFrame(rows)
    df.to_csv(MASTER, index=False, encoding="utf-8-sig")
    print(f"\nmaster: {MASTER}  ({len(df)} runs, IDs {df.ID.min()}-{df.ID.max()})")
    print("\nresumen por propuesta:")
    for idx, role in chosen:
        r = df[df.pareto_role == role].iloc[0]
        on = [f"{a.split('_')[1][:5]}[{r['on_'+a]}-{r['off_'+a]}]" for a in ACTS4 if r[f"act_{a}"]]
        print(f"  {role:22s}: {on or 'sin acciones'}")


if __name__ == "__main__":
    main()
