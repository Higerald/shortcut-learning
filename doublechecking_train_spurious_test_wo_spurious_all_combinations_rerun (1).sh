#!/usr/bin/env bash
set -euo pipefail
set -o monitor

# ===== Config =====
PY=python
SCRIPT=transformer_lr_ar_spurious_checking_v1.py

SEEDS=0,1,2,3,4
DLIST=2,4,6
HIDDEN=2,3,4
LAYERS=1,2
REGIMES=erm
NTEST=4000
LR=1e-3
BS=4096

GPU_LIST=(4 5 6)

# Conservative to stabilize CUDA init first
JOBS_PER_GPU=2
MIN_FREE_MB=16000
PICK_RETRY_DELAY=4
LAUNCH_SPREAD_DELAY=1.0
RETRY_802_ONCE=true
FAST_FAIL_SECONDS=20
MAX_CONCURRENT_JOBS=$(( ${#GPU_LIST[@]} * JOBS_PER_GPU ))

CORE_SCALE_LIST=("0.25" "0.50" "0.75" "1.00" "1.25")
# SIGMA_CORE_LIST=("0.05" "0.5" "1.5")
SIGMA_CORE_LIST=("1.5" "0.5" "0.05")
SIGMA_SPU_LIST=("0.05" "0.5" "1.5")
SIGMA_NOISE_LIST=("0.05" "0.5" "1.5")
B_LIST=("0.5" "1.0" "1.5")

# ===== Preflight =====
if ! command -v nvidia-smi >/dev/null; then
  echo "[FATAL] nvidia-smi not found." >&2; exit 1
fi

# Verify CUDA visible in this shell before scheduling hundreds of jobs.
python - <<'PY'
import torch, sys
ok = torch.cuda.is_available() and torch.cuda.device_count()>0
sys.exit(0 if ok else 3)
PY
if [[ $? -ne 0 ]]; then
  echo "[FATAL] CUDA not available to this shell (torch.cuda.is_available()==False or device_count==0)." >&2
  exit 3
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results_batch_${STAMP}"
mkdir -p "$OUTDIR"
echo "Writing outputs to: $OUTDIR"

PIDMAP_FILE="$(mktemp /tmp/gpu_pidmap.XXXXXX)"

# ===== Helpers =====
prune_dead() {
  local tmp; tmp="$(mktemp)"
  if [[ -s "$PIDMAP_FILE" ]]; then
    while read -r pid gpu; do
      if kill -0 "$pid" 2>/dev/null; then echo "$pid $gpu" >> "$tmp"; fi
    done < "$PIDMAP_FILE"
  fi
  mv -f "$tmp" "$PIDMAP_FILE"
}

gpu_live_counts() {
  prune_dead
  declare -A counts=()
  for g in "${GPU_LIST[@]}"; do counts["$g"]=0; done
  if [[ -s "$PIDMAP_FILE" ]]; then
    while read -r _pid gpu; do
      [[ " ${GPU_LIST[*]} " == *" $gpu "* ]] && counts["$gpu"]=$((counts["$gpu"]+1))
    done < "$PIDMAP_FILE"
  fi
  for g in "${GPU_LIST[@]}"; do echo "$g ${counts[$g]}"; done
}

total_running_jobs(){ jobs -pr | wc -l | tr -d ' '; }

wait_for_slot() {
  while true; do
    local running; running=$(total_running_jobs)
    if (( running < MAX_CONCURRENT_JOBS )); then break; fi
    sleep 5
  done
}

pick_gpu_balanced_mem() {
  while true; do
    mapfile -t count_lines < <(gpu_live_counts)
    declare -A free_map=()
    local free_csv
    free_csv=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    while IFS=',' read -r idx free; do
      idx="${idx// /}"; free="${free// /}"
      [[ -n "$idx" && -n "$free" ]] && free_map["$idx"]="$free"
    done <<< "$free_csv"

    local best_gpu=""; local best_cnt=999999; local best_free=-1
    for line in "${count_lines[@]}"; do
      local g c; g=$(awk '{print $1}' <<<"$line"); c=$(awk '{print $2}' <<<"$line")
      [[ -z "${free_map[$g]+x}" ]] && continue
      local f="${free_map[$g]:-0}"
      if (( c < JOBS_PER_GPU )) && [[ "$f" =~ ^[0-9]+$ ]] && (( f >= MIN_FREE_MB )); then
        if (( c < best_cnt )) || { (( c == best_cnt )) && (( f > best_free )); }; then
          best_gpu="$g"; best_cnt="$c"; best_free="$f"
        fi
      fi
    done

    if [[ -n "$best_gpu" ]]; then echo "$best_gpu"; return 0; fi
    sleep "$PICK_RETRY_DELAY"
  done
}

run() {
  local name="$1"; shift
  wait_for_slot
  local gpu; gpu="$(pick_gpu_balanced_mem)"

  local csv="$OUTDIR/${name}.csv"
  local log="$OUTDIR/${name}.log"

  echo ">>> [ENQUEUE] $name on GPU $gpu"
  { echo "COMMAND: $*"; echo "GPU: $gpu"; } | tee "$log"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    # Enable MPS for this child if daemon is present (best effort)
    export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps_pipe}
    export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/mps_log}
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d >/dev/null 2>&1 || true

    # tiny warm-up (optional)
    nvidia-smi -L >/dev/null 2>&1 || true

    exec "$@" --out-csv "$csv"
  ) 2>&1 | tee -a "$log" &
  local pid=$!
  echo "$pid $gpu" >> "$PIDMAP_FILE"

  (
    sleep "$FAST_FAIL_SECONDS"
    if kill -0 "$pid" 2>/dev/null; then exit 0; fi
    if $RETRY_802_ONCE && grep -q "Error 802: system not yet initialized" "$log"; then
      echo "[retry] Early CUDA init failure detected for $name; retrying once..." | tee -a "$log"
      sleep 6
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps_pipe}
        export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/mps_log}
        mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
        nvidia-cuda-mps-control -d >/dev/null 2>&1 || true
        nvidia-smi -L >/dev/null 2>&1 || true
        exec "$@" --out-csv "$csv"
      ) 2>&1 | tee -a "$log" &
      local pid2=$!
      echo "$pid2 $gpu" >> "$PIDMAP_FILE"
    fi
  ) &

  sleep "$LAUNCH_SPREAD_DELAY"
}

run_combo() {
  local TAG="$1"; local IDCFG="$2"; local OODCFG="$3"
  echo "=== Starting combo: $TAG ==="
  run "low_data_${TAG}_noSpuOOD"  "$PY" "$SCRIPT" --device cuda --seeds "$SEEDS" \
      --d-list "$DLIST" --head-hidden-list "$HIDDEN" --head-layers-list "$LAYERS" \
      --regimes "$REGIMES" --n-test "$NTEST" --epochs 500 --lr "$LR" --batch-size "$BS" \
      --id-cfg-json  "$IDCFG" --ood-cfg-json "$OODCFG" \
      --counts-by-group-train-json '{"(1,1)":60,"(-1,-1)":60,"(1,-1)":10,"(-1,1)":10}'
  run "med_data_${TAG}_noSpuOOD"  "$PY" "$SCRIPT" --device cuda --seeds "$SEEDS" \
      --d-list "$DLIST" --head-hidden-list "$HIDDEN" --head-layers-list "$LAYERS" \
      --regimes "$REGIMES" --n-test "$NTEST" --epochs 500 --lr "$LR" --batch-size "$BS" \
      --id-cfg-json  "$IDCFG" --ood-cfg-json "$OODCFG" \
      --counts-by-group-train-json '{"(1,1)":600,"(-1,-1)":600,"(1,-1)":60,"(-1,1)":60}'
  run "high_data_${TAG}_noSpuOOD" "$PY" "$SCRIPT" --device cuda --seeds "$SEEDS" \
      --d-list "$DLIST" --head-hidden-list "$HIDDEN" --head-layers-list "$LAYERS" \
      --regimes "$REGIMES" --n-test "$NTEST" --epochs 500 --lr "$LR" --batch-size "$BS" \
      --id-cfg-json  "$IDCFG" --ood-cfg-json "$OODCFG" \
      --counts-by-group-train-json '{"(1,1)":1200,"(-1,-1)":1200,"(1,-1)":200,"(-1,1)":200}'
}

# ===== Sweep =====
for CORE_SCALE in "${CORE_SCALE_LIST[@]}"; do
  for SIGC in "${SIGMA_CORE_LIST[@]}"; do
    for SPU in "${SIGMA_SPU_LIST[@]}"; do
      for NOISE in "${SIGMA_NOISE_LIST[@]}"; do
        for B in "${B_LIST[@]}"; do
          TAG="core${CORE_SCALE}_sigc${SIGC}_spu${SPU}_noise${NOISE}_B${B}"
          IDCFG=$(printf '{"core_scale": %s, "sigma_core": %s, "sigma_spu": %s, "sigma_noise": %s, "B": %s}' \
                          "$CORE_SCALE" "$SIGC" "$SPU" "$NOISE" "$B")
          OODCFG=$(printf '{"core_scale": %s, "sigma_core": %s, "sigma_spu": %s, "sigma_noise": %s, "B": 0.0}' \
                          "$CORE_SCALE" "$SIGC" "$SPU" "$NOISE")
          run_combo "$TAG" "$IDCFG" "$OODCFG"
        done
      done
    done
  done
done

wait
echo "All jobs completed in: $OUTDIR"
rm -f "$PIDMAP_FILE"
