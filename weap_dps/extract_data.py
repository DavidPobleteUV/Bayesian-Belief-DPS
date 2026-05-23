# -*- coding: utf-8 -*-
"""
extract_data.py — Copia los artefactos necesarios desde el repo del modelo a
`data_weap/` y prepara un template del input X tomando un run baseline del
zarr merged.

Idempotente: si los archivos ya existen y son más nuevos que el origen, no
los reemplaza.

Uso:
    python weap_dps/extract_data.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

# Acceso a constantes del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weap_dps.config_weap import (
    CKPT_PATH, MANIFEST_PATH, SCALERS_PATH, TRANSFORM_PARAMS_PATH,
    ZARR_TEMPLATE_PATH, DATA_DIR, MODEL_REPO,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [EXTRACT] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


import argparse
import re


def _resolve_best_ckpt(explicit: str | None = None) -> Path:
    """Resuelve el checkpoint a extraer.

    Prioridad:
      1. --checkpoint explícito.
      2. Mejor (menor val_loss) en MODEL_REPO/runs/iter01/best_model-*.ckpt.
      3. Cualquier best_model-*.ckpt bajo runs/ (recursivo), menor val_loss.
      4. Fallback histórico: MODEL_REPO/runs/best_model.ckpt.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = MODEL_REPO / p
        return p

    def _val_loss(path: Path) -> float:
        m = re.search(r"val_loss=([0-9]+\.[0-9]+)", path.name)
        return float(m.group(1)) if m else float("inf")

    for search_dir in [MODEL_REPO / "runs" / "iter01", MODEL_REPO / "runs"]:
        if not search_dir.exists():
            continue
        cands = sorted(search_dir.rglob("best_model-*.ckpt"), key=_val_loss)
        if cands:
            return cands[0]

    return MODEL_REPO / "runs" / "best_model.ckpt"


# SRC se construye en main() porque el ckpt se resuelve dinámicamente
def _build_src(ckpt_src: Path) -> dict:
    return {
        CKPT_PATH:             ckpt_src,
        MANIFEST_PATH:         MODEL_REPO / "data" / "variables_mlp_weekly_filtered.csv",
        SCALERS_PATH:          MODEL_REPO / "data" / "scalers_weap.npz",
        TRANSFORM_PARAMS_PATH: MODEL_REPO / "data" / "transform_params_weap.npz",
    }


def copy_if_newer(src: Path, dst: Path) -> bool:
    if not src.exists():
        logger.error("Fuente no existe: %s", src)
        return False
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        logger.info("Skip (igual o más nuevo): %s", dst.name)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("Copiado: %s → %s", src.name, dst)
    return True


def build_X_template(baseline_run_id: int = 0) -> None:
    """
    Extrae el X_filtered del run `baseline_run_id` del zarr del modelo
    y lo guarda como template (.npz). El template servirá de "esqueleto"
    para construir inputs sintéticos en el bridge: ya viene con los lags
    GW iniciales y las columnas climáticas/áreas/población del run base.
    """
    zarr_path = MODEL_REPO / "data" / "weap_weekly.zarr"
    if not zarr_path.exists():
        logger.warning("Zarr no encontrado en %s — saltando template.", zarr_path)
        return

    z = zarr.open_group(str(zarr_path), mode="r")
    if "X_filtered" not in z:
        logger.error("X_filtered no está en el zarr. Corre prepare_training.py primero.")
        return

    run_ids = z["run_ids"][:]
    idx = np.where(run_ids == baseline_run_id)[0]
    if len(idx) == 0:
        logger.error("run_id=%d no está en el zarr.", baseline_run_id)
        return
    slot = int(idx[0])
    X = z["X_filtered"][slot]               # (1872, 611)
    mask = z["mask"][slot] if "mask" in z else None
    feature_names = list(z.attrs.get("feature_names_filtered",
                                      z.attrs.get("feature_names", [])))

    ZARR_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ZARR_TEMPLATE_PATH,
        X=X.astype(np.float32),
        mask=mask if mask is not None else np.ones(X.shape[0], dtype=bool),
        feature_names=np.array(feature_names, dtype=object),
        baseline_run_id=baseline_run_id,
    )
    logger.info("Template guardado: %s  shape=%s", ZARR_TEMPLATE_PATH, X.shape)


def main():
    ap = argparse.ArgumentParser(description="Copia ckpt + scalers + manifest del modelo al DPS.")
    ap.add_argument("--checkpoint", default=None,
                    help="Ruta a un .ckpt específico (abs o relativa a MODEL_REPO). "
                         "Por defecto usa el mejor de runs/iter01 (menor val_loss).")
    ap.add_argument("--baseline_run_id", type=int, default=0,
                    help="run_id base para el X template.")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Repo modelo: %s", MODEL_REPO)
    logger.info("Destino: %s", DATA_DIR)

    ckpt_src = _resolve_best_ckpt(args.checkpoint)
    logger.info("Checkpoint fuente: %s", ckpt_src)
    SRC = _build_src(ckpt_src)

    n_copied = 0
    for dst, src in SRC.items():
        if copy_if_newer(src, dst):
            n_copied += 1

    logger.info("Archivos copiados: %d/%d", n_copied, len(SRC))
    build_X_template(baseline_run_id=args.baseline_run_id)
    logger.info("Done.")


if __name__ == "__main__":
    main()
