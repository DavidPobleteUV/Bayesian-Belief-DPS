# Scripts heredados de WEAP_HydroMLP_RecursiveGW

Diez scripts de junio de 2026 que vivían en la raíz del repo de entrenamiento
pero cuya lógica es de **decisión, no de emulación**: calculan costos en CLP,
resuelven la asignación de agua entre fuentes y calibran objetivos J4/J6, y para
ello leían `town_source_cost_mapping.csv` de este repo.

Se movieron aquí para que la frontera entre repos sea la misma que rige entre
`WEAP_2_ZARR` y `WEAP_HydroMLP_RecursiveGW`:

| repo | responsabilidad |
|---|---|
| `WEAP_2_ZARR` | generar los zarr desde WEAP y unirlos |
| `WEAP_HydroMLP_RecursiveGW` | normalizar, entrenar y evaluar el emulador |
| `Bayesian-Belief-DPS` | objetivos, costos, asignación entre fuentes, políticas |

## Estado

**Son material histórico, no el camino vigente.** `weap_dps/waterfall_alloc.py`
—la cascada de despacho que usa el DPS— es el descendiente directo de
`eval_v33_waterfall.py`, y su registro de localidades es un port recortado de
`train_v5_allocation.build_registry`. La versión vigente difiere en dos puntos
que importan:

- el **orden de despacho se deriva** de las tarifas (`merit_order()`) en vez de
  estar escrito a mano;
- el **acuerdo de reasignación participa** de la cascada con su tope de 25 L/s,
  en lugar de quedar excluido y con sus enlaces en cero.

La calibración de J4 que producía `calibrate_j4_waterfall.py` también quedó
obsoleta: se recalculó sobre el emulador `iter1_fix2050` y la tabla por número
de acciones pasó a ser plana (ver `weap_dps/config_weap.py`).

## Dependencias

El grupo es autocontenido: todos importan de `train_v5_allocation` y nada fuera
de esta carpeta los importa. Varios apuntan a rutas de datos antiguas
(`data/weap_weekly.zarr`, checkpoints anteriores) que ya no existen, de modo que
**no corren tal cual**. Se conservan por trazabilidad de cómo se llegó a la
formulación actual, no para reutilizarse.
