#!/usr/bin/env bash
# run_production_4variants.sh — full production DPS for the 4 model variants:
#   v2 (native), v2.3 (waterfall), v3 (native), v3.3 (waterfall)
# 5 seeds x 8000 evals each (pop=100). Per-variant J4 calibration (re-derived).
# Resumable: skips any (variant,seed) whose .dat already exists.
set -u
cd "$(dirname "$0")"
PY="venv_DPS/Scripts/python.exe"
V2="../WEAP_HydroMLP_RecursiveGW/runs/iter07_v2_clean/best_model-epoch=021-val_loss=0.0623.ckpt"
V3="../WEAP_HydroMLP_RecursiveGW/runs/iter07_v3_clean/best_model-epoch=011-val_loss=0.0638.ckpt"
EVALS=8000; POP=100
SEEDS="42 123 456 789 1024"
# per-variant J4 calibration (calibrate_j4_waterfall.py)
CAL_v2=1.149; CAL_v2_3=1.189; CAL_v3=1.032; CAL_v3_3=1.184
OUT=runs_weap/prod; mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/master.log"; }

# name ckpt wf cal
run_variant() {
  local name="$1" ckpt="$2" wf="$3" cal="$4"
  for s in $SEEDS; do
    local out="$OUT/pareto_${name}_seed${s}.dat"
    if [ -f "$out" ]; then log "SKIP $name seed=$s (exists)"; continue; fi
    log "START $name seed=$s (WF=$wf J4cal=$cal)"
    DPS_CKPT="$ckpt" DPS_WATERFALL="$wf" DPS_J4_CAL="$cal" \
      "$PY" weap_dps/main_par_weap.py --evaluations "$EVALS" --population "$POP" \
        --seed "$s" --workers 1 --output "$out" > "$OUT/${name}_seed${s}.log" 2>&1
    if [ -f "$out" ]; then log "DONE  $name seed=$s"; else log "FAIL  $name seed=$s"; fi
  done
}

log "===== PRODUCTION 4 variants x 5 seeds x ${EVALS} evals ====="
run_variant "v2"   "$V2" 0 "$CAL_v2"
run_variant "v2_3" "$V2" 1 "$CAL_v2_3"
run_variant "v3"   "$V3" 0 "$CAL_v3"
run_variant "v3_3" "$V3" 1 "$CAL_v3_3"
log "===== ALL PRODUCTION RUNS DONE ====="
touch "$OUT/ALL_DONE"
