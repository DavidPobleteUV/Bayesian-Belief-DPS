# -*- coding: utf-8 -*-
"""
weap_dps — Bridge entre WEAP-HydroMLP y Standard DPS (Quilimari).

Namespace que aísla todo el código del case study Quilimari (Chile) del
código original del paper Bayesian DPS en `src/`. No modifica `src/`.

Componentes:
  - config_weap       : Constantes y rutas del case study.
  - extract_data      : Copia ckpt + scalers + manifest desde el repo del modelo.
  - mlp_surrogate     : Wrapper del WEAP-HydroMLP con rollout anual.
  - climate_sampler   : Genera series weekly de precip+temp por subcuenca.
  - demand_builder    : Aplica crecimiento poblacional y escalado de áreas.
  - action_translator : Mapea políticas DPS (3 bin + 3 cont) → input MLP.
  - cost_calculator   : Calcula los 5 objetivos J1..J5.
  - pipe_simulation_weap : Loop temporal anual con decisión adaptive (Opción B).
  - pipe_problem_weap : Wrapper Platypus para optimización multiobjetivo.
  - main_par_weap     : Entry point de la optimización.
"""

__version__ = "0.1.0"
