# -*- coding: utf-8 -*-
"""
pareto_to_runids.py — Exporta soluciones del frente de Pareto a runs WEAP.

Loop iterativo (active learning surrogate-assisted):
  1. Optimización DPS → frente de Pareto en MLP space.
  2. Selecciona 7 soluciones representativas (5 extremos + 2 balanceadas).
  3. Para cada (solución × clima), evalúa la policy NN sobre el template del
     clima y genera un schedule anual de las 5 acciones binarias.
  4. Escribe:
      a) Un CSV por solución (`policy_iter<N>_<rid>.csv`) compatible con
         ReadFromFile() de WEAP.
      b) Un master CSV (`RunIDs_Q_pareto_iter<N>.csv`) con metadata por run.
      c) Un sidecar JSON con policy params, objectives MLP, etc.
  5. Estos archivos se copian a la carpeta WEAP y se corren con
     `WEAP_2_ZARR/src/pipeline/run_pipeline.py` (requiere extensión de
     `weap_runner.py` para leer schedules — ver nota al final de USAGE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import (
    DATA_DIR, ZARR_TEMPLATE_PATH, CKPT_PATH,
    SPIN_UP_YEARS, DECISION_YEARS, WARMUP_WEEKS, WEEKS_PER_YEAR,
    ACTION_NAMES_BINARY, ACTION_NAMES_QUANTITY,
    GCM_LIST,
)
from weap_dps.mlp_surrogate import MLPSurrogate
from weap_dps.action_translator import (
    policy_output_to_actions, build_action_col_idx,
)
from weap_dps.pipe_simulation_weap import PipeWEAP

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [PARETO_EXPORT] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# El MLP conoce las 4 acciones vigentes (desal costera, desal completa,
# pozo a 5km, acuerdo), así que no hay acciones "desconocidas" que fijar en 0.
ACTIONS_UNKNOWN_TO_MLP = []

OBJECTIVE_NAMES = [
    "J1_neg_storage",     # negado: min(-J1) = max(J1)
    "J2_unmet",
    "J3_neg_agri",
    "J4_cost",
    "J5_weeks_failure",
]


# ─── 1. Carga y selección de soluciones ─────────────────────────────────────

@dataclass
class ParetoSolution:
    variables: np.ndarray
    objectives: np.ndarray
    role: str = ""              # "extremo_J1", "balanced_1", etc.


def load_pareto(dat_path: Path) -> list[ParetoSolution]:
    with open(dat_path, "rb") as f:
        data = pickle.load(f)
    sols = []
    for vars_, objs in data["result"]:
        sols.append(ParetoSolution(
            variables=np.asarray(vars_, dtype=float),
            objectives=np.asarray(objs, dtype=float),
        ))
    logger.info("Loaded %d Pareto solutions from %s", len(sols), dat_path)
    return sols, data


def select_representative_solutions(
    pareto: list[ParetoSolution],
    n_balanced: int = 2,
) -> list[ParetoSolution]:
    """
    Selecciona 5 extremos (uno por objetivo, minimizando cada Jk) +
    n_balanced balanceadas (clustering k-means sobre las restantes).
    """
    if len(pareto) < 5 + n_balanced:
        raise ValueError(f"Pareto tiene {len(pareto)} soluciones, "
                         f"necesito al menos {5 + n_balanced}")

    objs = np.array([s.objectives for s in pareto])    # (N, 5)

    # 5 extremos
    selected_idx = []
    for k in range(5):
        idx = int(np.argmin(objs[:, k]))
        if idx not in selected_idx:
            pareto[idx].role = f"extremo_J{k+1}"
            selected_idx.append(idx)
        else:
            # Si el mismo punto es mejor en 2 dimensiones, busca el segundo mejor
            sorted_idx = np.argsort(objs[:, k])
            for cand in sorted_idx:
                if cand not in selected_idx:
                    pareto[int(cand)].role = f"extremo_J{k+1}"
                    selected_idx.append(int(cand))
                    break

    # n_balanced balanceadas: clustering sobre las no-extremas
    remaining = [i for i in range(len(pareto)) if i not in selected_idx]
    if remaining and n_balanced > 0:
        # Normalizar objs por rango antes del clustering
        scaler_range = np.maximum(objs.max(0) - objs.min(0), 1e-9)
        objs_norm = (objs[remaining] - objs.min(0)) / scaler_range
        n_clust = min(n_balanced, len(remaining))
        km = KMeans(n_clusters=n_clust, n_init=10, random_state=42).fit(objs_norm)
        # Para cada cluster, tomar el punto más cercano al centroide
        for c in range(n_clust):
            mask = km.labels_ == c
            if not mask.any():
                continue
            dists = np.linalg.norm(objs_norm[mask] - km.cluster_centers_[c], axis=1)
            local_idx = np.argmin(dists)
            global_idx = remaining[np.where(mask)[0][local_idx]]
            pareto[global_idx].role = f"balanced_{c+1}"
            selected_idx.append(global_idx)

    out = [pareto[i] for i in selected_idx]
    for s in out:
        logger.info("  selected: %s → objs=%s", s.role,
                    np.array2string(s.objectives, precision=3))
    return out


# ─── 2. Generación del schedule anual ───────────────────────────────────────

def evaluate_policy_to_schedule(
    solution: ParetoSolution,
    surrogate: MLPSurrogate,
    pipe: PipeWEAP,
    climate_template: np.ndarray | None = None,
) -> np.ndarray:
    """
    Corre la policy NN sobre el horizonte y devuelve action_history.
    Shape: (n_years, 6) con orden = ACTION_NAMES_BINARY + ACTION_NAMES_QUANTITY.

    Si climate_template es None, usa el X_template default cargado por pipe.
    """
    policy_fn = pipe._build_policy_from_params(solution.variables)
    X = climate_template if climate_template is not None else pipe.X_template

    result = surrogate.rollout_with_policy(
        X_template=X,
        policy_fn=policy_fn,
        n_years=DECISION_YEARS,
        action_col_idx=pipe.action_col_idx,
        spin_up_years=SPIN_UP_YEARS,
    )
    return result["actions_history"]


def schedule_to_weap_csv(
    action_history: np.ndarray,
    start_year: int,
    output_path: Path,
    include_q_values: bool = False,
    hydro_end_year: int = 2060,
) -> None:
    """
    Genera CSV en formato WEAP `$Columns = ...` con 2 filas por AÑO HIDROLÓGICO
    (4/2/YYYY y 4/1/(YYYY+1) mismo valor) para comportamiento step bajo
    interpolación Average. El año-agua inicia el 2-abril (water_year_month=4).

    Solo escribe columnas binarias act_* + las 2 acciones desconocidas al
    MLP (siempre 0). Las cantidades q_* se omiten porque el modelo WEAP
    usa capacidades canónicas (L_01 = 100 l/s, etc.). Para activar
    capacidades continuas habría que modificar el modelo WEAP — ver nota
    en USAGE.md.

    Si include_q_values=True, agrega también las columnas q_* (útil para
    debug y para análisis post-WEAP).

    `hydro_end_year` DEBE llegar al fin de la simulación WEAP (2060). Estuvo en
    2050 y ese desfase es exactamente el defecto que invalidó la iteración
    anterior: las expresiones de capacidad son
    `If(ReadFromFile(pf, col) > 0, CAP_ALTA, CAP_BAJA)`, de modo que sin dato la
    capacidad cae a CAP_BAJA y la acción deja de entregar agua mientras las
    columnas act_*/q_* de X la siguen declarando activa. Fueron 520 de 1.716
    semanas del periodo de decisión con entrada y salida en contradicción.
    """
    n_years = action_history.shape[0]
    if include_q_values:
        cols = ACTION_NAMES_BINARY + ACTION_NAMES_QUANTITY + ACTIONS_UNKNOWN_TO_MLP
    else:
        cols = ACTION_NAMES_BINARY + ACTIONS_UNKNOWN_TO_MLP

    rows = []
    for y in range(n_years):
        year = start_year + y
        row_values = {}
        # Binarias
        for i, name in enumerate(ACTION_NAMES_BINARY):
            row_values[name] = int(action_history[y, i])
        # Cantidades (las columnas q empiezan después de las binarias)
        if include_q_values:
            n_bin = len(ACTION_NAMES_BINARY)
            for i, name in enumerate(ACTION_NAMES_QUANTITY):
                row_values[name] = float(action_history[y, n_bin + i])
        # Acciones desconocidas → siempre 0
        for name in ACTIONS_UNKNOWN_TO_MLP:
            row_values[name] = 0

        # AÑO HIDROLÓGICO (inicia 2-abril, water_year_month=4 day=2): la decisión del
        # año `year` rige 4/2/year -> 4/1/(year+1), valor constante (step) en la ventana.
        for date_str in [f"4/2/{year}", f"4/1/{year + 1}"]:
            row = {"Date": date_str, **row_values}
            rows.append(row)

    # Extender (HOLD del estado final de decisión) hasta el fin del año hidrológico que
    # TERMINA en `hydro_end_year` (= 4/1/hydro_end_year, fin del año-agua = corrida WEAP).
    # Las acciones son step irreversibles: se mantienen tras la última decisión.
    last_year = start_year + n_years - 1          # último año-agua con decisión (e.g. 2047)
    for year in range(last_year + 1, hydro_end_year):   # year+1 alcanza hydro_end_year
        for date_str in [f"4/2/{year}", f"4/1/{year + 1}"]:
            rows.append({"Date": date_str, **row_values})        # row_values = estado final

    df = pd.DataFrame(rows, columns=["Date"] + cols)

    # Escribir con header WEAP `$Columns = ...`
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("$Columns = " + ",".join(df.columns) + "\n")
        df.to_csv(f, index=False, header=False, lineterminator="\n")

    logger.info("Schedule CSV: %s  (%d años, %d filas)",
                output_path, n_years, len(rows))


# ─── 3. Master CSV (RunIDs) y metadata sidecar ──────────────────────────────

def build_master_csv(
    runs: list[dict],
    output_path: Path,
) -> None:
    """
    Master CSV compatible con `WEAP_2_ZARR/data/RunIDs_Q_lhs_extreme.csv`,
    con una columna extra `policy_schedule_csv` que apunta al CSV de
    schedule por run.
    """
    df = pd.DataFrame(runs)
    # Orden de columnas recomendado para diff visual con RunIDs_Q_lhs_extreme
    cols_order = [
        "ID",
        # Set vigente (K=4): prorrateo_shac/cuenca fueron ELIMINADAS del catálogo
        # WEAP y se incorporó acuerdo. Debe calzar con ACTION_NAMES_BINARY y con
        # las columnas de data/RunIDs_Q_full.csv del proyecto WEAP_2_ZARR.
        "act_desalacion_costera", "act_desalacion_completa",
        "act_nuevo_pozo_a_5km", "act_acuerdo",
        "GCM", "SSP",
        "drought_severity", "drought_duration", "drought_start_year",
        "temperature_delta", "drought_severity_mode",
        "Demanda_Agro", "Demanda_Poblacion",
        "policy_schedule_csv",
        "source_pareto", "pareto_role",
    ]
    for c in cols_order:
        if c not in df.columns:
            df[c] = ""
    df = df[cols_order]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Master CSV: %s  (%d runs)", output_path, len(df))


def write_metadata(
    iteration: int,
    pareto_path: Path,
    selected_sols: list[ParetoSolution],
    climates: list[tuple[str, str]],
    output_path: Path,
) -> None:
    ckpt_hash = hashlib.md5(CKPT_PATH.read_bytes()).hexdigest()[:12] \
                if CKPT_PATH.exists() else "MISSING"
    meta = {
        "iteration":       iteration,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "pareto_source":   str(pareto_path),
        "mlp_ckpt_hash":   ckpt_hash,
        "n_solutions":     len(selected_sols),
        "n_climates":      len(climates),
        "climates":        [{"gcm": g, "ssp": s} for g, s in climates],
        "selected_solutions": [
            {
                "role":       s.role,
                "objectives": s.objectives.tolist(),
                "variables":  s.variables.tolist(),
            }
            for s in selected_sols
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Metadata: %s", output_path)


# ─── 4. Orchestrator ────────────────────────────────────────────────────────

DEFAULT_DEMAND_AGRO = "Sin cambio en Areas Regadas"
DEFAULT_DEMAND_POB  = "Crecimiento anual regular: 2%"


def export_iteration(
    pareto_dat: Path,
    iteration: int,
    output_dir: Path,
    start_id: int,
    climates: list[tuple[str, str]],
    n_balanced: int = 2,
    start_year: int = 2027,   # primera decisión DPS = año-agua 2027 (verificado vs time[])
    include_q_values: bool = True,
) -> None:
    """
    Punto de entrada principal.
    """
    iter_dir = output_dir / f"iter_{iteration:02d}"
    policies_dir = iter_dir / "Policies"
    policies_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar Pareto y seleccionar 7 representativas
    pareto, raw = load_pareto(pareto_dat)
    selected = select_representative_solutions(pareto, n_balanced=n_balanced)

    # 2. Cargar surrogate y pipe (reusables entre runs)
    pipe = PipeWEAP(template_path=ZARR_TEMPLATE_PATH)
    surrogate = pipe.surrogate
    logger.info("Surrogate cargado. n_sols=%d × n_climas=%d = %d runs",
                len(selected), len(climates), len(selected) * len(climates))

    # 3. Generar CSVs por (sol × clima)
    runs_master = []
    current_id = start_id
    for sol in selected:
        action_hist = evaluate_policy_to_schedule(sol, surrogate, pipe)
        for gcm, ssp in climates:
            policy_csv_name = f"policy_iter{iteration:02d}_{current_id:04d}_{sol.role}_{gcm}.csv"
            policy_csv_path = policies_dir / policy_csv_name
            schedule_to_weap_csv(
                action_hist, start_year=start_year,
                output_path=policy_csv_path,
                include_q_values=include_q_values,
            )

            # Año-1 acciones como representación summary en el master CSV
            act_year1 = {
                name: int(action_hist[0, i])
                for i, name in enumerate(ACTION_NAMES_BINARY)
            }
            runs_master.append({
                "ID": current_id,
                # act_year1 ya trae las 4 binarias vigentes desde
                # ACTION_NAMES_BINARY; no hay acciones extra que fijar en 0.
                **act_year1,
                "GCM": gcm,
                "SSP": ssp,
                "drought_severity":   "",
                "drought_duration":   "",
                "drought_start_year": "",
                "temperature_delta":  "",
                "drought_severity_mode": "",
                "Demanda_Agro":      DEFAULT_DEMAND_AGRO,
                "Demanda_Poblacion": DEFAULT_DEMAND_POB,
                "policy_schedule_csv": f"Policies/{policy_csv_name}",
                "source_pareto":     pareto_dat.name,
                "pareto_role":       sol.role,
            })
            current_id += 1

    # 4. Master CSV + metadata
    master_csv = iter_dir / f"RunIDs_Q_pareto_iter{iteration:02d}.csv"
    build_master_csv(runs_master, master_csv)
    write_metadata(iteration, pareto_dat, selected, climates,
                    iter_dir / "metadata.json")

    logger.info("=" * 60)
    logger.info("Iteración %d exportada:", iteration)
    logger.info("  Directorio: %s", iter_dir)
    logger.info("  IDs: %d–%d", start_id, current_id - 1)
    logger.info("  Policies CSV: %d archivos", len(runs_master))
    logger.info("=" * 60)
    logger.info("")
    logger.info("Próximos pasos:")
    logger.info("  1. Copiar %s/Policies/ a:", iter_dir)
    logger.info("     C:\\Users\\David\\Documents\\WEAP Areas\\Quilimari_WEAP_MODFLOW_RDM\\Policies\\")
    logger.info("  2. Copiar %s a:", master_csv.name)
    logger.info("     C:\\Users\\David\\Documents\\GitHub_DPL\\WEAP_2_ZARR\\data\\")
    logger.info("  3. Agregar la ruta del master CSV en config.yaml → runids_lhs_files")
    logger.info("  4. Correr WEAP_2_ZARR pipeline con --run_ids %d..%d", start_id, current_id - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pareto",      type=Path, required=True,
                   help="Path al .dat del Pareto generado por main_par_weap.py")
    p.add_argument("--iteration",   type=int, required=True,
                   help="Número de iteración (1, 2, 3, ...)")
    p.add_argument("--start_id",    type=int, default=None,
                   help="ID inicial. Default: 2000 + (iteration-1)*100. "
                        "Convención: 0-999 factorial, 1000-1999 LHS/sintéticos, "
                        "2000+ Pareto (100 IDs por iteración).")
    p.add_argument("--output_dir",  type=Path, default=DATA_DIR / "exports")
    p.add_argument("--start_year",  type=int, default=2027,
                   help="Primer año-agua de decisión DPS (default 2027, verificado vs time[])")
    p.add_argument("--no_q",        action="store_true",
                   help="No incluir columnas q_* en el CSV de schedule")
    args = p.parse_args()

    if args.start_id is None:
        args.start_id = 2000 + (args.iteration - 1) * 100
        print(f"[pareto_to_runids] --start_id default → {args.start_id} "
              f"(iter {args.iteration}, bloque 2000+{args.iteration-1}*100)")

    # 3 climas default: los 3 GCMs más diversos
    climates = [
        ("MPI-ESM1-2-LR", "ssp585"),
        ("ACCESS-CM2",    "ssp585"),
        ("GFDL-ESM4",     "ssp585"),
    ]

    export_iteration(
        pareto_dat=args.pareto,
        iteration=args.iteration,
        output_dir=args.output_dir,
        start_id=args.start_id,
        climates=climates,
        n_balanced=2,
        start_year=args.start_year,
        include_q_values=not args.no_q,
    )


if __name__ == "__main__":
    main()
