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
def _build_src(ckpt_src: Path, data_dir: Path | None = None) -> dict:
    """Fuentes a copiar al DPS.

    `data_dir` = carpeta del modelo donde viven scalers/transform/manifest
    FILTRADO. Debe ser la MISMA con la que se entrenó el checkpoint: mezclar un
    ckpt nuevo con scalers viejos desnormaliza mal y en silencio.
    Por defecto MODEL_REPO/data (layout antiguo).
    """
    d = data_dir or (MODEL_REPO / "data")
    return {
        CKPT_PATH:             ckpt_src,
        MANIFEST_PATH:         d / "variables_mlp_weekly_filtered.csv",
        SCALERS_PATH:          d / "scalers_weap.npz",
        TRANSFORM_PARAMS_PATH: d / "transform_params_weap.npz",
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


def build_X_template(baseline_run_id: int = 0,
                     zarr_path: Path | None = None,
                     manifest_path: Path | None = None) -> None:
    """
    Extrae el X_filtered del run `baseline_run_id` del zarr del modelo
    y lo guarda como template (.npz). El template servirá de "esqueleto"
    para construir inputs sintéticos en el bridge: ya viene con los lags
    GW iniciales y las columnas climáticas/áreas/población del run base.

    IMPORTANTE: el DataModule sub-selecciona las columnas de X_filtered por el
    manifest (x_idx) antes de entrenar, así que el modelo espera len(x_idx)
    columnas, NO todas las de X_filtered. Si el template se guarda sin ese
    sub-set, mlp_surrogate falla con "X shape mismatch". Por eso aquí se aplica
    el mismo filtro (pasando --manifest).
    """
    zarr_path = zarr_path or (MODEL_REPO / "data" / "weap_weekly.zarr")
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
    X = z["X_filtered"][slot]               # (T, n_cols_filtered)
    mask = z["mask"][slot] if "mask" in z else None
    feature_names = list(z.attrs.get("feature_names_filtered",
                                      z.attrs.get("feature_names", [])))

    # ── sub-set por manifest (igual que el DataModule) ──
    # Además se guardan los índices de salida (gw/surface) EN EL ESPACIO DE
    # Y_filtered, calculados con la MISMA función que usa el DataModule. Sin
    # esto el DPS los re-derivaba del orden de filas del manifest, que es un
    # espacio distinto (685 filas vs 677 columnas de Y_filtered) → IndexError.
    target_names = list(z.attrs.get("target_names_filtered",
                                    z.attrs.get("target_names", [])))
    extra = {}
    if manifest_path and Path(manifest_path).exists():
        import sys as _sys
        _sys.path.insert(0, str(MODEL_REPO / "src"))
        try:
            from rdm_mlp.utils.manifest import load_manifest, build_indices
            x_idx, y_idx, ar_idx = build_indices(feature_names, target_names,
                                                 load_manifest(str(manifest_path)))
            if x_idx is not None and len(x_idx) != X.shape[1]:
                logger.info("Sub-seleccionando X por manifest: %d → %d columnas",
                            X.shape[1], len(x_idx))
                sel = np.asarray(x_idx, dtype=int)
                X = X[:, sel]
                feature_names = [feature_names[i] for i in sel]

            gw_idx = np.asarray(ar_idx, dtype=int)
            y_all = y_idx if y_idx is not None else range(len(target_names))
            surf_idx = np.array([i for i in y_all if i not in set(gw_idx.tolist())],
                                dtype=int)
            extra = dict(
                gw_idx_filt=gw_idx,
                surface_idx_filt=surf_idx,
                target_names_filtered=np.array(target_names, dtype=object),
            )
            logger.info("Índices de salida: n_gw=%d  n_surface=%d  (espacio Y_filtered=%d)",
                        len(gw_idx), len(surf_idx), len(target_names))
        except Exception as exc:      # noqa: BLE001
            logger.warning("No se pudo aplicar el sub-set del manifest (%s). "
                           "El template puede no calzar con el modelo.", exc)

    ZARR_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ZARR_TEMPLATE_PATH,
        X=X.astype(np.float32),
        mask=mask if mask is not None else np.ones(X.shape[0], dtype=bool),
        feature_names=np.array(feature_names, dtype=object),
        baseline_run_id=baseline_run_id,
        **extra,
    )
    logger.info("Template guardado: %s  shape=%s", ZARR_TEMPLATE_PATH, X.shape)


def main():
    ap = argparse.ArgumentParser(description="Copia ckpt + scalers + manifest del modelo al DPS.")
    ap.add_argument("--checkpoint", default=None,
                    help="Ruta a un .ckpt específico (abs o relativa a MODEL_REPO). "
                         "Por defecto usa el mejor de runs/iter01 (menor val_loss).")
    ap.add_argument("--baseline_run_id", type=int, default=0,
                    help="run_id base para el X template.")
    ap.add_argument("--zarr", default=None,
                    help="Zarr del modelo (rel. a MODEL_REPO o absoluto). "
                         "Ej: data/_v3_900/weap_weekly_merged.zarr")
    ap.add_argument("--manifest", default=None,
                    help="Manifest FILTRADO del modelo, para sub-seleccionar X "
                         "igual que el DataModule (evita 'X shape mismatch'). "
                         "Ej: data/_v3_900/variables_mlp_weekly_filtered.csv")
    args = ap.parse_args()

    def _resolve(p):
        if not p:
            return None
        q = Path(p)
        return q if q.is_absolute() else (MODEL_REPO / q)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Repo modelo: %s", MODEL_REPO)
    logger.info("Destino: %s", DATA_DIR)

    ckpt_src = _resolve_best_ckpt(args.checkpoint)
    logger.info("Checkpoint fuente: %s", ckpt_src)
    # scalers/transform/manifest deben salir de la MISMA carpeta con la que se
    # entrenó el ckpt. Si se pasó --manifest, se usa su carpeta.
    data_dir = _resolve(args.manifest).parent if args.manifest else None
    if data_dir:
        logger.info("Carpeta de artefactos del modelo: %s", data_dir)
    SRC = _build_src(ckpt_src, data_dir)

    n_copied = 0
    for dst, src in SRC.items():
        if copy_if_newer(src, dst):
            n_copied += 1

    logger.info("Archivos copiados: %d/%d", n_copied, len(SRC))
    build_X_template(baseline_run_id=args.baseline_run_id,
                     zarr_path=_resolve(args.zarr),
                     manifest_path=_resolve(args.manifest))
    logger.info("Done.")


if __name__ == "__main__":
    main()
