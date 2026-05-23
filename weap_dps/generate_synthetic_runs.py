# -*- coding: utf-8 -*-
"""
generate_synthetic_runs.py — Genera filas sintéticas de "todas las acciones
activas" con distintos climas (con/sin drought) para complementar el set de
runs WEAP de la iteración.

No requiere policy NN ni schedule CSV — acciones constantes en formato LHS:
todas las binarias = 1 durante todo el horizonte. WEAP usa las capacidades
canónicas de cada acción.

Output: data_weap/exports/iter_<N>/RunIDs_Q_all_on_iter<N>.csv

Uso:
    python weap_dps/generate_synthetic_runs.py --iteration 1 --start_id 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Forzar UTF-8 en stdout para que PowerShell renderice caracteres especiales
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import DATA_DIR


# 10 escenarios cubriendo el gradiente completo (cataclysm → normal).
# Niveles: 2 cataclysm (sev=0.95) + 2 muy secos (sev=0.90) + 2 secos (sev=0.70)
#          + 2 leves (sev=0.30) + 2 normales (sin drought).
# Variamos GCM, start_year y ΔT para mayor diversidad climática.
SYNTHETIC_SCENARIOS = [
    # ── Cataclysm (precip al 5%) ──
    {"GCM": "CanESM5",       "SSP": "ssp585",
     "drought_severity": 0.95, "drought_duration": 30, "drought_start_year": 2025,
     "temperature_delta": 3, "drought_severity_mode": "extreme",
     "tag": "CanESM5_cataclysm30y_dT3"},
    {"GCM": "MPI-ESM1-2-LR", "SSP": "ssp585",
     "drought_severity": 0.95, "drought_duration": 20, "drought_start_year": 2030,
     "temperature_delta": 3, "drought_severity_mode": "extreme",
     "tag": "MPI_megadrought20y_dT3"},

    # ── Muy secos (precip al 10%) ──
    {"GCM": "MPI-ESM1-2-LR", "SSP": "ssp585",
     "drought_severity": 0.90, "drought_duration": 25, "drought_start_year": 2025,
     "temperature_delta": 3, "drought_severity_mode": "extreme",
     "tag": "MPI_drought25y_dT3"},
    {"GCM": "ACCESS-CM2",    "SSP": "ssp585",
     "drought_severity": 0.90, "drought_duration": 20, "drought_start_year": 2030,
     "temperature_delta": 2, "drought_severity_mode": "extreme",
     "tag": "ACCESS_drought20y_dT2"},

    # ── Secos (precip al 30%) ──
    {"GCM": "GFDL-ESM4",     "SSP": "ssp585",
     "drought_severity": 0.70, "drought_duration": 20, "drought_start_year": 2030,
     "temperature_delta": 2, "drought_severity_mode": "extreme",
     "tag": "GFDL_drought20y_sev070_dT2"},
    {"GCM": "CanESM5",       "SSP": "ssp585",
     "drought_severity": 0.70, "drought_duration": 15, "drought_start_year": 2035,
     "temperature_delta": 1, "drought_severity_mode": "extreme",
     "tag": "CanESM5_drought15y_sev070_dT1"},

    # ── Leves (precip al 70%) ──
    {"GCM": "MPI-ESM1-2-LR", "SSP": "ssp585",
     "drought_severity": 0.30, "drought_duration": 10, "drought_start_year": 2030,
     "temperature_delta": 1, "drought_severity_mode": "extreme",
     "tag": "MPI_drought10y_sev030_leve"},
    {"GCM": "ACCESS-CM2",    "SSP": "ssp585",
     "drought_severity": 0.30, "drought_duration":  5, "drought_start_year": 2035,
     "temperature_delta": 0, "drought_severity_mode": "extreme",
     "tag": "ACCESS_drought5y_sev030_leve"},

    # ── Normales (sin drought) ──
    {"GCM": "MPI-ESM1-2-LR", "SSP": "ssp585",
     "drought_severity": "", "drought_duration": "", "drought_start_year": "",
     "temperature_delta": "", "drought_severity_mode": "",
     "tag": "MPI_normal"},
    {"GCM": "ACCESS-CM2",    "SSP": "ssp585",
     "drought_severity": "", "drought_duration": "", "drought_start_year": "",
     "temperature_delta": "", "drought_severity_mode": "",
     "tag": "ACCESS_normal"},
]

DEFAULT_DEMAND_AGRO = "Sin cambio en Areas Regadas"
DEFAULT_DEMAND_POB  = "Crecimiento anual regular: 2%"


def build_synthetic_rows(start_id: int) -> list[dict]:
    """Construye filas con todas las 5 acciones activas y los 4 escenarios."""
    rows = []
    for i, scen in enumerate(SYNTHETIC_SCENARIOS):
        rows.append({
            "ID": start_id + i,
            "act_desalacion_costera":   1,
            "act_desalacion_completa":  1,
            "act_prorrateo_shac":       1,
            "act_prorrateo_cuenca":     1,
            "act_nuevo_pozo_a_5km":     1,
            "GCM": scen["GCM"],
            "SSP": scen["SSP"],
            "drought_severity":     scen["drought_severity"],
            "drought_duration":     scen["drought_duration"],
            "drought_start_year":   scen["drought_start_year"],
            "temperature_delta":    scen["temperature_delta"],
            "drought_severity_mode": scen["drought_severity_mode"],
            "Demanda_Agro":      DEFAULT_DEMAND_AGRO,
            "Demanda_Poblacion": DEFAULT_DEMAND_POB,
            "policy_schedule_csv": "",   # acciones constantes desde el master CSV
            "source_pareto":     "synthetic_all_on",
            "pareto_role":       f"all_on_{scen['tag']}",
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, required=True,
                   help="Número de iteración (1, 2, ...). Debe coincidir con el del Pareto export.")
    p.add_argument("--start_id",  type=int, required=True,
                   help="ID inicial. Debe estar DESPUÉS del último ID del Pareto export para no chocar.")
    p.add_argument("--output_dir", type=Path, default=DATA_DIR / "exports")
    args = p.parse_args()

    iter_dir = args.output_dir / f"iter_{args.iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    out_csv = iter_dir / f"RunIDs_Q_all_on_iter{args.iteration:02d}.csv"

    rows = build_synthetic_rows(args.start_id)
    df = pd.DataFrame(rows)

    # Orden de columnas igual al master CSV de pareto_to_runids
    cols_order = [
        "ID",
        "act_desalacion_costera", "act_desalacion_completa",
        "act_prorrateo_shac", "act_prorrateo_cuenca", "act_nuevo_pozo_a_5km",
        "GCM", "SSP",
        "drought_severity", "drought_duration", "drought_start_year",
        "temperature_delta", "drought_severity_mode",
        "Demanda_Agro", "Demanda_Poblacion",
        "policy_schedule_csv",
        "source_pareto", "pareto_role",
    ]
    df = df[cols_order]
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"Generado: {out_csv}")
    print(f"  {len(rows)} runs sintéticos con todas las acciones activas")
    print(f"  IDs: {args.start_id}–{args.start_id + len(rows) - 1}")
    print("=" * 60)
    print(df[["ID", "pareto_role", "GCM", "drought_severity",
              "drought_duration", "temperature_delta"]].to_string(index=False))
    print()
    print("Próximo paso:")
    print(f"  1. Copiar {out_csv.name} a WEAP_2_ZARR/data/")
    print(f"  2. Agregar al config.yaml → runids_lhs_files:")
    print(f"     - data/{out_csv.name}")
    print(f"  3. Lanzar el pipeline con --run_ids {args.start_id}..{args.start_id + len(rows) - 1}")


if __name__ == "__main__":
    main()
