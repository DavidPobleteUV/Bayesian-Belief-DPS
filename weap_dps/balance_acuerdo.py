# -*- coding: utf-8 -*-
"""
balance_acuerdo.py — balance neto del acuerdo de reasignación PARA EL AGRICULTOR
y sensibilidad a la tarifa.

El acuerdo transfiere agua desde riego a agua potable: el agricultor pierde
producción y recibe un pago. Se aísla el efecto comparando runs con SOLO el
acuerdo activo contra su par sin ninguna acción (mismo GCM, SSP, demanda agrícola
y demanda poblacional), de modo que la diferencia no arrastre otras obras.

Se separan dos precios que NO son el mismo número:

  · `tarifa_agricultor`  — lo que efectivamente recibe quien cede el agua. Es el
    que decide si el agricultor participa (balance neto).
  · `tarifa_total`       — lo que paga el operador urbano = tarifa_agricultor +
    costo de transacción/conducción que no llega al predio. Es el que define la
    posición del acuerdo en el orden de mérito del despacho.

Referencia externa: Ávila et al. (2025, JWRPM 151(4):05025003) estiman para el
Aconcagua el costo de oportunidad del agua de riego en 0.85 USD/m³ (rango
sensibilizado 0.35–1.35), esto es 300–1158 CLP/m³ al tipo de cambio 857.66 que
usa ese trabajo. En su diseño la compensación IGUALA el costo de oportunidad, de
modo que el agricultor queda indiferente; aquí se reporta además cuánta prima
sobre ese punto de indiferencia hace falta para sostener adhesión voluntaria.

Uso:
    python weap_dps/balance_acuerdo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weap_dps.config_weap import (BASE_YEAR, DISCOUNT_RATE, UNIT_COST_BY_SOURCE,
                                  USD_CLP_RATE)
from weap_dps.cost_calculator import j3_agricultural_value
from weap_dps.waterfall_alloc import merit_order

H = (Path(r"C:\Users\David\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW") /
     "data" / "_v3_900_fix2050" / "weap_weekly_merged.zarr")
M = Path(r"C:\Users\David\Documents\GitHub_DPL\WEAP_2_ZARR\data\RunIDs_Q_full.csv")
W0, WPY = 676, 52
OUT = Path("results")

# Candidatos a evaluar: (tarifa que recibe el agricultor, costo que no le llega).
# (1500, 300) es la configuración adoptada; el resto queda como sensibilidad.
CANDIDATOS = [(2500, 0), (1000, 300), (1200, 300), (1500, 300), (1700, 300),
              (2000, 300)]
# Referencias de Ávila et al. (2025), CLP/m3 al tipo de cambio 857.66 del paper
AVILA = {"bajo (0.35 USD/m3)": 300, "central (0.85 USD/m3)": 729,
         "alto (1.35 USD/m3)": 1158}


def pares() -> pd.DataFrame:
    """Pérdida de producción y volumen cedido (ambos en NPV) por par de runs."""
    Z = zarr.open_group(str(H), mode="r")
    tn = list(Z.attrs["target_names"])
    zr = np.asarray(Z["run_ids"][:]).astype(int)
    m = pd.read_csv(M, encoding="utf-8-sig")
    A = ["desalacion_costera", "desalacion_completa", "nuevo_pozo_a_5km", "acuerdo"]
    s = m[m.ID.isin(set(zr.tolist()))]
    solo = s[(s.act_acuerdo == 1) & (s[["act_" + a for a in A[:3]]].sum(axis=1) == 0)]
    base = s[s[["act_" + a for a in A]].sum(axis=1) == 0]
    k = ["GCM", "SSP", "Demanda_Agro", "Demanda_Poblacion"]
    par = solo.merge(base, on=k, suffixes=("_ac", "_base"))

    # El agua del acuerdo sale de los nodos DemAGRO: son esos enlaces los que
    # miden el volumen efectivamente cedido por el riego.
    j_ac = [i for i, n in enumerate(tn)
            if n.startswith("AP_TransmissionLinks__") and "DemAGRO" in n and "_to_" in n]
    print(f"enlaces del acuerdo: {len(j_ac)}   descuento r={DISCOUNT_RATE:.0%}, "
          f"base {BASE_YEAR}")

    filas = []
    for _, r in par.iterrows():
        ia = int(np.where(zr == r.ID_ac)[0][0])
        ib = int(np.where(zr == r.ID_base)[0][0])
        Ya, Yb = np.nan_to_num(Z["Y"][ia]), np.nan_to_num(Z["Y"][ib])
        v_ac = j3_agricultural_value(Ya, tn, decision_start_week=W0)
        v_bs = j3_agricultural_value(Yb, tn, decision_start_week=W0)
        if not (np.isfinite(v_ac) and np.isfinite(v_bs)):
            continue
        vol_sem = np.maximum(Ya[W0:, j_ac], 0).sum(axis=1)          # m3/semana
        n_y = len(vol_sem) // WPY
        anual = vol_sem[:n_y * WPY].reshape(n_y, WPY).sum(axis=1)
        # Volumen DESCONTADO: multiplicado por cualquier tarifa da el pago NPV,
        # de modo que barrer tarifas no exige releer el zarr.
        vol_npv = float(sum(v / (1 + DISCOUNT_RATE) ** t for t, v in enumerate(anual)))
        filas.append({"ID": int(r.ID_ac), "vol_Mm3": anual.sum() / 1e6,
                      "vol_npv_m3": vol_npv, "perdida": v_bs - v_ac})
    return pd.DataFrame(filas)


def evaluar(d: pd.DataFrame, tarifa: float) -> dict:
    pago = d.vol_npv_m3 * tarifa
    neto = pago - d.perdida
    razon = pago / d.perdida
    return {"tarifa": tarifa, "pago_med": np.median(pago),
            "neto_med": np.median(neto), "neto_MUSD": np.median(neto) / USD_CLP_RATE / 1e6,
            "razon_med": np.median(razon), "razon_p25": np.percentile(razon, 25),
            "pct_positivo": 100 * (neto > 0).mean()}


def main() -> int:
    d = pares()
    print(f"pares evaluados: {len(d)}   volumen cedido mediano: "
          f"{np.median(d.vol_Mm3):.2f} Mm3 (33 años)")
    print(f"pérdida de producción mediana: {np.median(d.perdida):.3e} CLP NPV "
          f"({np.median(d.perdida) / USD_CLP_RATE / 1e6:.2f} MUSD)\n")

    eq = d.perdida / d.vol_npv_m3
    print("=== Tarifa de equilibrio (iguala pérdida y pago) ===")
    print(f"  mediana {np.median(eq):,.0f} CLP/m3   p25 {np.percentile(eq, 25):,.0f}"
          f"   p75 {np.percentile(eq, 75):,.0f}   p90 {np.percentile(eq, 90):,.0f}")
    print("  referencia Aconcagua (Ávila et al. 2025):")
    for k, v in AVILA.items():
        print(f"    {k:22s} {v:6,.0f} CLP/m3  -> cubre el "
              f"{100 * (eq <= v).mean():4.0f} % de los pares")

    print("\n=== Balance del agricultor por tarifa ===")
    print(f"{'recibe':>8} {'total':>7} {'pago NPV':>11} {'neto NPV':>11} {'MUSD':>7} "
          f"{'razón':>7} {'>0':>6}  orden de mérito del acuerdo")
    filas = []
    for t_ag, t_tx in CANDIDATOS:
        e = evaluar(d, t_ag)
        total = t_ag + t_tx
        costos = dict(UNIT_COST_BY_SOURCE, Acuerdo=float(total))
        orden = merit_order(costos)
        pos = orden.index("Acuerdo") + 1
        e.update({"tarifa_agricultor": t_ag, "costo_transaccion": t_tx,
                  "tarifa_total": total, "posicion_merito": pos,
                  "orden": " < ".join(orden)})
        filas.append(e)
        print(f"{t_ag:8,.0f} {total:7,.0f} {e['pago_med']:11.3e} {e['neto_med']:11.3e} "
              f"{e['neto_MUSD']:7.2f} {e['razon_med']:7.2f} {e['pct_positivo']:5.0f}% "
              f"  {pos}º de {len(orden)}:  {' < '.join(orden)}")

    print("\n=== Lectura ===")
    print("  La razón pago/pérdida es el margen del agricultor: 1.00 es el punto")
    print("  de indiferencia de Ávila et al., donde la compensación solo repone lo")
    print("  perdido. Una adhesión voluntaria necesita margen sobre ese punto, y el")
    print("  '>0' indica en qué fracción de climas el acuerdo le conviene.")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(OUT / "balance_acuerdo.csv", index=False,
                               encoding="utf-8-sig")
    d.to_csv(OUT / "balance_acuerdo_pares.csv", index=False, encoding="utf-8-sig")
    print(f"\n  tabla: {OUT / 'balance_acuerdo.csv'}")
    print(f"  pares: {OUT / 'balance_acuerdo_pares.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
