#!/usr/bin/env bash
# run_robust.sh — Robust DPS (ensamble climate × demand, mean + λ·std) sobre los
# 2 MLP base v2 y v3. Preset C: 3 seeds × 4000 evals × (5 clima × 3 demanda = 15
# escenarios). Los 6 runs (2 modelos × 3 seeds) corren EN PARALELO (OMP=1, 1 core
# c/u). ~20 h. Reanudable: salta .dat existentes.
set -u
cd "$(dirname "$0")"
PY="venv_DPS/Scripts/python.exe"
V2="../WEAP_HydroMLP_RecursiveGW/runs/iter07_v2_clean/best_model-epoch=021-val_loss=0.0623.ckpt"
V3="../WEAP_HydroMLP_RecursiveGW/runs/iter07_v3_clean/best_model-epoch=011-val_loss=0.0638.ckpt"
EVALS=4000; POP=100; NCLIM=5; LAM=1.0
SEEDS="42 123 456"
CAL_v2=1.149; CAL_v3=1.032            # calibración J4 nativa por modelo
OUT=runs_weap/robust; mkdir -p "$OUT"
log(){ echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/master.log"; }

launch(){ # name ckpt cal seed
  local name="$1" ckpt="$2" cal="$3" s="$4"
  local out="$OUT/pareto_${name}_seed${s}.dat"
  if [ -f "$out" ]; then log "SKIP $name seed=$s (existe)"; return; fi
  log "LAUNCH $name seed=$s"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    DPS_CKPT="$ckpt" DPS_WATERFALL=0 DPS_J4_CAL="$cal" \
    "$PY" weap_dps/main_robust_weap.py --evaluations "$EVALS" --population "$POP" \
      --seed "$s" --n_climate "$NCLIM" --lam "$LAM" --output "$out" \
      > "$OUT/${name}_seed${s}.log" 2>&1 &
}

log "===== ROBUST DPS  2 modelos × 3 seeds × ${EVALS} evals × ${NCLIM}clima·3dem (λ=${LAM}) ====="
for s in $SEEDS; do launch v2 "$V2" "$CAL_v2" "$s"; launch v3 "$V3" "$CAL_v3" "$s"; done
log "6 procesos lanzados en paralelo — esperando..."
wait
log "===== ROBUST DPS DONE ====="; touch "$OUT/ALL_DONE"
