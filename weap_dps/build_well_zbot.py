# -*- coding: utf-8 -*-
"""
build_well_zbot.py — Genera la geometría estática de pozos que necesita J6.

J6 (salinidad costera) ya NO se lee de `WF_SalinityFactor`: esa variable se sacó
del entrenamiento porque es una función DETERMINISTA del Z_value (interfaz agua
dulce/salada del SWI2) y de la geometría del pozo. El MLP predice Z_value; la
salinidad se reconstruye con:

    z_bot = TOP - profundidad_del_pozo            (estático, este archivo)
    SalinityFactor = 0                    si z_bot > Z + Z_trans   (dulce)
                     1                    si z_bot < Z - Z_trans   (salada)
                     (Z + Z_trans - z_bot) / (2·Z_trans)  en la transición

Salida: data_weap/reference/well_zbot.csv  (WellName, TOP, WellDepth_m, z_bot)
Es pequeño y estático → va al repo, así el DPS queda self-contained.

Uso:
    python weap_dps/build_well_zbot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
W2Z = PROJECT_ROOT.parent / "WEAP_2_ZARR"
OUT = PROJECT_ROOT / "data_weap" / "reference" / "well_zbot.csv"


def main() -> None:
    sys.path.insert(0, str(W2Z / "src"))
    from groundwater.well_factors import load_wells   # noqa: E402

    wells = load_wells(W2Z / "data" / "reference" / "wells_data.xlsx")
    top = np.load(W2Z / "data" / "reference" / "top.npy")

    # MF_ROW/MF_COL vienen 1-based desde WEAP
    r = wells["MF_ROW"].astype(int).to_numpy() - 1
    c = wells["MF_COL"].astype(int).to_numpy() - 1
    wells["TOP"] = top[r, c]
    wells["z_bot"] = wells["TOP"] - wells["WellDepth_m"].astype(float)

    out = wells[["WellName", "TOP", "WellDepth_m", "z_bot"]].copy()
    out = out.dropna(subset=["WellName", "z_bot"]).drop_duplicates("WellName")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"pozos: {len(out)}  →  {OUT}")
    print(f"z_bot: min={out['z_bot'].min():.1f}  max={out['z_bot'].max():.1f} m")
    cost = out[out["WellName"].str.contains(r"APU_Q09|APR_Q09|Pozo_Costero", regex=True)]
    print(f"de esos, costeros (APR/APU Q09 + Pozo_Costero_DOH): {len(cost)}")


if __name__ == "__main__":
    main()
