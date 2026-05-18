# Instrucciones para Claude

## Idioma

**Responde siempre en español latino neutro.** NO uses formas argentinas / voseo.

| Usar (tú) | NO usar (vos) |
|---|---|
| dime, dímelo | decime, decímelo |
| avísame | avisame |
| quieres | querés |
| tienes | tenés |
| puedes | podés |
| hazlo, haz | hacelo, hacé |
| pégalo, pégame | pegalo, pegame |
| confírmame | confirmame |
| ve a, anda a | andá a |
| mira, mírate | fijate |

Evitar también: "che", "dale", "boludo", "loco", "posta", "garpa", etc.

Mantener este estilo en TODAS las conversaciones del proyecto.

## Contexto del proyecto

Tres repositorios hermanos en `C:\Users\David\Documents\GitHub_DPL\`:

- **`WEAP_2_ZARR`** — pipeline WEAP/MODFLOW que simula la cuenca Quilimari y exporta a un Zarr store.
- **`WEAP_HydroMLP_RecursiveGW`** — modelo surrogate MLP recursivo entrenado sobre los runs WEAP.
- **`Bayesian-Belief-DPS`** — optimización multiobjetivo Standard DPS con el bridge `weap_dps/` que conecta el MLP al optimizador NSGA-II.

El flujo iterativo (active learning):

```
WEAP runs → entrena MLP → optimiza DPS → Pareto front
              ↑                              ↓
              └──── retrain con nuevos runs ─┘
```

## Convenciones

- **Encoding**: UTF-8 en todos los archivos. Los scripts Python entry-point hacen `sys.stdout.reconfigure(encoding="utf-8")` al inicio para que PowerShell renderice correctamente caracteres como `═`, `→`, `✓`, acentos.
- **Imports relativos**: el namespace del bridge es `weap_dps/`. Los scripts hacen `sys.path.insert(0, parent)` para que `from weap_dps.X import Y` funcione al correr como script.
- **Paquete editable**: `rdm_mlp` se instala con `pip install -e ../WEAP_HydroMLP_RecursiveGW` en el venv del DPS para que los cambios en el modelo se propaguen sin reinstalar.

## Convenciones de objetivos (DPS)

Los 5 objetivos del optimizador, definidos en `weap_dps/cost_calculator.py`:

| # | Nombre | Dirección | Unidad |
|---|---|---|---|
| J1 | GW Storage (min temporal) | maximizar | m³ |
| J2 | Unmet AP acumulado | minimizar | m³ (sumado sobre el horizonte) |
| J3 | Valor agrícola acumulado | maximizar | CLP |
| J4 | Costo de abastecimiento | minimizar | CLP |
| J5 | Semanas en falla | minimizar | count (> 100 m³/sem) |

Para NSGA-II, J1 y J3 se almacenan negados internamente (todo a minimizar). Al cargar para visualización, se restauran los signos en `plot_pareto.py`.

## Estilo de respuesta

- Conciso. Sin pad de cortesía innecesario.
- Tablas para comparativas.
- Bloques de código solo cuando aportan.
- Al cerrar una tarea, indicar siguiente paso concreto.
