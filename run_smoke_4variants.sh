#!/usr/bin/env bash
# run_smoke_4variants.sh — quick smoke (1 seed, 2000 evals) of the 4 model variants
# through DPS, in sequence:  v2 (native), v2.3 (waterfall), v3 (native), v3.3 (waterfall).
# Validates the full 4-variant pipeline before any long production run.
set -u
cd "$(dirname "$0")"
PY="venv_DPS/Scripts/python.exe"
V2="../WEAP_HydroMLP_RecursiveGW/runs/iter07_v2_clean/best_model-epoch=021-val_loss=0.0623.ckpt"
V3="../WEAP_HydroMLP_RecursiveGW/runs/iter07_v3_clean/best_model-epoch=011-val_loss=0.0638.ckpt"
EVALS=2000; POP=100; SEED=42
mkdir -p runs_weap/smoke
log() { echo "$(date +%H:%M:%S) $*" | tee -a runs_weap/smoke/master.log; }

# per-variant J4 calibration (re-derived via calibrate_j4_waterfall.py)
CAL_v2=1.149; CAL_v2_3=1.189; CAL_v3=1.032; CAL_v3_3=1.184

# variant  ckpt  waterfall  j4cal
run_variant() {
  local name="$1" ckpt="$2" wf="$3" cal="$4"
  local out="runs_weap/smoke/pareto_${name}.dat"
  log "START $name  (WF=$wf  J4cal=$cal)  -> $out"
  DPS_CKPT="$ckpt" DPS_WATERFALL="$wf" DPS_J4_CAL="$cal" \
    "$PY" weap_dps/main_par_weap.py --evaluations "$EVALS" --population "$POP" \
      --seed "$SEED" --workers 1 --output "$out" > "runs_weap/smoke/${name}.log" 2>&1
  if [ -f "$out" ]; then log "DONE  $name  ($(stat -c%s "$out") bytes)"; else log "FAIL  $name (no output)"; fi
}

log "===== smoke 4 variants: evals=$EVALS pop=$POP seed=$SEED ====="
run_variant "v2"   "$V2" 0 "$CAL_v2"
run_variant "v2_3" "$V2" 1 "$CAL_v2_3"
run_variant "v3"   "$V3" 0 "$CAL_v3"
run_variant "v3_3" "$V3" 1 "$CAL_v3_3"
log "===== ALL SMOKE VARIANTS DONE ====="
touch runs_weap/smoke/ALL_DONE
