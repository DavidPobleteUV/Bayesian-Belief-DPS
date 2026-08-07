# -*- coding: utf-8 -*-
"""
build_train_subset.py — Zarr REDUCIDO para correr el DPS sin mover el dataset completo.

`scenario_builder` necesita el zarr de entrenamiento para armar el ensamble
climático, pero solo lee tres cosas:

    Z.attrs["feature_names"]      nombres de las features
    Z["run_ids"]                  ids de run
    Z["X"]                        features CRUDAS (sin normalizar)

No toca `Y`, ni `X_filtered`, ni `Y_filtered` — que son el 90% del peso. Además
solo usa el run baseline (0) y los N runs climáticos que elige por precipitación.
Este script extrae exactamente eso: el zarr pasa de ~6 GB a ~30 MB, y viaja junto
a los otros artefactos en `data_weap/`.

Los runs climáticos se PRE-SELECCIONAN aquí con el mismo criterio que usa
`pick_climate_runs` (repartidos seco→húmedo por precipitación total), así el
ensamble que arme el servidor es idéntico al que armaría con el dataset completo.

Uso:
    python weap_dps/build_train_subset.py                 # n_climate=5
    python weap_dps/build_train_subset.py --n_climate 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import TRAIN_ZARR_PATH, DATA_DIR       # noqa: E402
from weap_dps.scenario_builder import SUBCUENCAS                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=None,
                    help="Zarr completo (default: el que resuelve config_weap).")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "train_subset.zarr")
    ap.add_argument("--n_climate", type=int, default=5,
                    help="Cuántos runs climáticos incluir (el DPS podrá usar hasta este número).")
    ap.add_argument("--baseline", type=int, default=0, help="run_id base (demanda/área/población).")
    args = ap.parse_args()

    src = args.source or TRAIN_ZARR_PATH
    print(f"origen : {src}")
    Z = zarr.open_group(str(src), mode="r")
    feat = list(Z.attrs["feature_names"])
    rids = np.array(Z["run_ids"][:]).astype(int)
    print(f"         {len(rids)} runs | X{Z['X'].shape}")

    # ── elegir los runs climáticos igual que pick_climate_runs ──
    pcols = [feat.index(f"Precipitation__{s}") for s in SUBCUENCAS
             if f"Precipitation__{s}" in feat]
    val = np.where(rids >= 0)[0]
    pp = Z["X"].oindex[:, :, pcols]
    tot = np.nansum(pp[val], axis=(1, 2))
    ok = tot > 0
    val, tot = val[ok], tot[ok]
    order = np.argsort(tot)
    pick = np.linspace(0, len(order) - 1, args.n_climate).astype(int)
    climate = [int(rids[val[order[p]]]) for p in pick]
    print(f"climas : {climate}  (seco→húmedo por precipitación total)")

    keep = sorted(set([args.baseline] + climate))
    slots = [int(np.where(rids == r)[0][0]) for r in keep]
    print(f"runs a copiar: {keep}")

    # ── escribir el zarr reducido ──
    if args.out.exists():
        shutil.rmtree(args.out)
    dst = zarr.open_group(str(args.out), mode="w")
    X = np.stack([Z["X"][s] for s in slots]).astype(np.float32)
    dst.create_array("X", shape=X.shape, dtype="float32", chunks=(1, X.shape[1], X.shape[2]))
    dst["X"][:] = X
    dst.create_array("run_ids", shape=(len(keep),), dtype="int32")
    dst["run_ids"][:] = np.array(keep, dtype=np.int32)
    for k, v in Z.attrs.items():                 # feature_names y demás metadatos
        dst.attrs[k] = v
    dst.attrs["subset_of"] = str(src)
    dst.attrs["subset_runs"] = keep
    dst.attrs["subset_climate_runs"] = climate

    mb = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"\nescrito: {args.out}")
    print(f"         X{X.shape}  —  {mb:.1f} MB")
    print(f"\nEn el servidor lo encuentra solo: config_weap lo busca primero en "
          f"data_weap/train_subset.zarr.\nOverride manual: $env:DPS_TRAIN_ZARR")


if __name__ == "__main__":
    main()
