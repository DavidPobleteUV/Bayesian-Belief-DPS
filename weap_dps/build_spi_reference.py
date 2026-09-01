# -*- coding: utf-8 -*-
"""
build_spi_reference.py — ajusta las gammas del SPI sobre el registro histórico.

El SPI estandariza una acumulación de precipitación contra una CLIMATOLOGÍA. La
elección de esa climatología es una decisión de método, no un detalle:

  · Contra el propio futuro de cada run, el índice queda centrado en cero por
    construcción y un mundo permanentemente seco marcaría SPI ~ 0. La política
    perdería justamente la señal que necesita para decidir si construir.
  · Contra el registro observado, un SPI de -1.5 significa lo mismo en todos los
    escenarios y es lo que un operador podría calcular en la realidad.

Se usa el registro de WEAP_2_ZARR/results/training_data/historico_1985_2019,
31 años-agua (1989-2020), precipitación MEDIA DE LA CUENCA sobre las 6
subcuencas: el SPI es un índice de cuenca, no de subcuenca.

El período incluye la megasequía 2010-2019 (166.8 mm/año contra 250.0 del
período previo, un déficit del 33%). Incluirla baja la media de referencia a
223.2 mm/año y por tanto ATENÚA los SPI negativos futuros. Se incluye a
propósito: es clima observado, no una anomalía que convenga excluir, y es el
registro que un operador tendría hoy en la mano.

La gamma se ajusta a la serie de ACUMULACIONES, no a la precipitación semanal.
Una ventana de 52 semanas ya integra el ciclo anual completo, de modo que la
serie está desestacionalizada por construcción y basta una gamma por ventana
(el SPI clásico sobre datos mensuales necesita doce, una por mes calendario).

Salida: data_weap/reference/spi_gamma.json — pequeño y estático, va al repo.

Uso:
    python weap_dps/build_spi_reference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HIST = (Path(__file__).resolve().parents[2] / "WEAP_2_ZARR" / "results" /
        "training_data" / "historico_1985_2019" / "weap_weekly.zarr")
OUT = Path(__file__).resolve().parents[1] / "data_weap" / "reference" / "spi_gamma.json"
VENTANAS = {"spi_12": 52, "spi_24": 104}


def rolling_sum(v: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.insert(v.astype(float), 0, 0.0))
    out = np.full(len(v), np.nan)
    out[n - 1:] = c[n:] - c[:-n]
    return out


def main() -> int:
    if not HIST.exists():
        raise SystemExit(f"No existe el registro histórico: {HIST}")
    Z = zarr.open_group(str(HIST), mode="r")
    f = list(Z.attrs["feature_names"])
    sub = [i for i, n in enumerate(f) if n.startswith("Precipitation__Subcuenca_")]
    if not sub:
        raise SystemExit("El zarr histórico no tiene columnas de precipitación.")
    pp = np.nan_to_num(Z["X"][0])[:, sub].mean(axis=1)      # media de cuenca, mm/sem
    n_anios = len(pp) / 52
    print(f"registro: {len(pp)} semanas ({n_anios:.1f} años-agua), "
          f"{pp.sum() / n_anios:.1f} mm/año, {len(sub)} subcuencas promediadas")

    ref = {"fuente": str(HIST), "n_semanas": int(len(pp)),
           "mm_anio_medio": float(pp.sum() / n_anios),
           "n_subcuencas": len(sub), "ventanas": {}}
    for nom, w in VENTANAS.items():
        acc = rolling_sum(pp, w)
        acc = acc[np.isfinite(acc)]
        # floc=0: la gamma del SPI se ancla en cero, no se desplaza. Sin esto el
        # ajuste puede correr el origen y el índice deja de ser comparable con
        # los SPI publicados.
        a, loc, scale = stats.gamma.fit(acc[acc > 0], floc=0.0)
        # Fracción de ceros: con 52 y 104 semanas es 0 en esta cuenca, pero se
        # guarda para que la transformación sea la mixta del SPI formal si
        # alguna vez se usa una ventana corta.
        q0 = float((acc <= 0).mean())
        ref["ventanas"][nom] = {
            "semanas": w, "forma": float(a), "escala": float(scale),
            "prob_cero": q0, "n": int(len(acc)),
            "acc_media": float(acc.mean()), "acc_p10": float(np.percentile(acc, 10)),
            "acc_p90": float(np.percentile(acc, 90)),
        }
        # Verificación: el SPI del propio registro debe dar media ~0 y sd ~1.
        spi = stats.norm.ppf(np.clip(q0 + (1 - q0) * stats.gamma.cdf(
            acc, a, loc=0.0, scale=scale), 1e-6, 1 - 1e-6))
        print(f"  {nom}: ventana {w:3d} sem  forma={a:6.3f} escala={scale:8.2f}  "
              f"acumulada media={acc.mean():7.1f} mm  "
              f"| SPI del registro: media={spi.mean():+.3f} sd={spi.std():.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nguardado: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
