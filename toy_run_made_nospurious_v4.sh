#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
PY=python
SCRIPT=toy_lr_made_no_spurious.py
SEEDS=0,1,2,3,4
DLIST=16,64,256
HIDDEN=64,256
LAYERS=1,2
REGIMES=erm,reweight,downsample,upsample
NTEST=4000
LR=1e-4
BS=64
EPOCHS=5000   # set to 100 / 500 / 5000 as desired

# Sweep class separation via core_scale (means at ±core_scale)
CLASS_SEP_LIST=("0.25" "0.50" "0.75" "1.00" "1.25")

# Optional: core feature noise
SIGMA_CORE_LIST=("0.5" "1.0" "1.5")

VAL_FRAC=0.2
SELECT_METRIC=val_acc_overall   # or val_acc_worst

# ========= GPU detection =========
# Priority:
# 1) Respect user-specified CUDA_VISIBLE_DEVICES if set
# 2) Else query nvidia-smi
# 3) Else run on CPU (no parallelism)

declare -a GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  # Parse lines like: GPU 0: ...
  mapfile -t GPU_IDS < <(nvidia-smi -L | awk -F'[ :]' '{print $2}')
fi

NUM_GPUS=${#GPU_IDS[@]}
if (( NUM_GPUS > 0 )); then
  echo "Detected GPUs: ${GPU_IDS[*]}"
  DEVICE=cuda
else
  echo "No GPUs detected. Falling back to CPU."
  DEVICE=cpu
  NUM_GPUS=1  # run serially when on CPU
fi

# ========= Outputs =========
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results_made_mlp_batch_${STAMP}"
mkdir -p "$OUTDIR"
echo "Writing outputs to: $OUTDIR"

# ========= Helpers =========

# Limit concurrency to NUM_GPUS. Requires bash 5+ for 'wait -n'.
_running_jobs=0
_next_gpu_idx=0

wait_for_slot() {
  if (( _running_jobs >= NUM_GPUS )); then
    # Wait for any job to finish before launching another
    wait -n
    _running_jobs=$((_running_jobs - 1))
  fi
}

pick_gpu() {
  # Returns chosen GPU id (or "cpu") in global var _CHOSEN_GPU
  if (( ${#GPU_IDS[@]} > 0 )); then
    local idx=$((_next_gpu_idx % NUM_GPUS))
    _CHOSEN_GPU="${GPU_IDS[$idx]}"
    _next_gpu_idx=$((_next_gpu_idx + 1))
  else
    _CHOSEN_GPU="cpu"
  fi
}

run() {
  local name="$1"; shift
  local csv="$OUTDIR/${name}.csv"
  local log="$OUTDIR/${name}.log"

  wait_for_slot
  pick_gpu

  if [[ "$_CHOSEN_GPU" == "cpu" ]]; then
    echo ">>> Running: $name on CPU"
    echo "COMMAND: $*" | tee "$log"
    "$@" --out-csv "$csv" > >(
      tee -a "$log"
    ) 2>&1 &
  else
    echo ">>> Running: $name on GPU $_CHOSEN_GPU"
    echo "COMMAND: $*" | tee "$log"
    CUDA_VISIBLE_DEVICES="$_CHOSEN_GPU" "$@" --out-csv "$csv" > >(
      tee -a "$log"
    ) 2>&1 &
  fi

  echo "  -> CSV: $csv"
  echo "  -> Log: $log"
  echo
  _running_jobs=$((_running_jobs + 1))
}

# Ensure we reap children on Ctrl-C
trap 'echo "Stopping..."; kill 0' INT TERM

# ========= Main =========
for SEP in "${CLASS_SEP_LIST[@]}"; do
  for SIGH in "${SIGMA_CORE_LIST[@]}"; do
    TAG="sep${SEP}_sig${SIGH}"
    IDCFG="{\"core_scale\": ${SEP}, \"sigma_core\": ${SIGH}}"
    OODCFG="{\"core_scale\": ${SEP}, \"sigma_core\": ${SIGH}}"

    # Low-data
    run "low_data_${TAG}_${EPOCHS}ep_nospurious" \
      $PY $SCRIPT \
        --device $DEVICE \
        --seeds $SEEDS \
        --d-list $DLIST \
        --head-hidden-list $HIDDEN \
        --head-layers-list $LAYERS \
        --regimes $REGIMES \
        --n-test $NTEST --epochs $EPOCHS --lr $LR --batch-size $BS \
        --val-from-id-frac $VAL_FRAC --select-metric $SELECT_METRIC \
        --id-cfg-json  "$IDCFG" \
        --ood-cfg-json "$OODCFG" \
        --counts-by-group-train-json '{"(1,1)":60,"(-1,-1)":60,"(1,-1)":10,"(-1,1)":10}'

    # Medium-data
    run "med_data_${TAG}_${EPOCHS}ep_nospurious" \
      $PY $SCRIPT \
        --device $DEVICE \
        --seeds $SEEDS \
        --d-list $DLIST \
        --head-hidden-list $HIDDEN \
        --head-layers-list $LAYERS \
        --regimes $REGIMES \
        --n-test $NTEST --epochs $EPOCHS --lr $LR --batch-size $BS \
        --val-from-id-frac $VAL_FRAC --select-metric $SELECT_METRIC \
        --id-cfg-json  "$IDCFG" \
        --ood-cfg-json "$OODCFG" \
        --counts-by-group-train-json '{"(1,1)":600,"(-1,-1)":600,"(1,-1)":60,"(-1,1)":60}'

    # High-data
    run "high_data_${TAG}_${EPOCHS}ep_nospurious" \
      $PY $SCRIPT \
        --device $DEVICE \
        --seeds $SEEDS \
        --d-list $DLIST \
        --head-hidden-list $HIDDEN \
        --head-layers-list $LAYERS \
        --regimes $REGIMES \
        --n-test $NTEST --epochs $EPOCHS --lr $LR --batch-size $BS \
        --val-from-id-frac $VAL_FRAC --select-metric $SELECT_METRIC \
        --id-cfg-json  "$IDCFG" \
        --ood-cfg-json "$OODCFG" \
        --counts-by-group-train-json '{"(1,1)":1200,"(-1,-1)":1200,"(1,-1)":200,"(-1,1)":200}'
  done
done

# Drain remaining jobs
wait
echo "All jobs completed!"






# #!/usr/bin/env bash
# set -euo pipefail

# # ---------- Config ----------
# PY=python
# SCRIPT=toy_lr_made.py   # <- best-checkpoint selection & 100 epochs
# SEEDS=0,1,2,3,4
# DLIST=16,64,256
# HIDDEN=64,256
# LAYERS=1,2
# REGIMES=erm,reweight,downsample,upsample
# NTEST=4000
# LR=1e-4
# BS=64
# #DEVICE=cpu                      # MLP/MADE are light; swap to cuda if you like
# DEVICE=cuda                      # MLP/MADE are light; swap to cuda if you like

# # Sweep class separation via core_scale (means at ±core_scale)
# # CLASS_SEP_LIST=("0.25" "0.50" "1.00")
# CLASS_SEP_LIST=("0.25" "0.50" "0.75" "1.00" "1.25")

# # (Optional) core feature noise; keep default 0.15 if you only want to vary separation
# # SIGMA_CORE_LIST=("0.15")
# SIGMA_CORE_LIST=("0.5" "1.0" "1.5")

# # Validation from ID test
# VAL_FRAC=0.2
# SELECT_METRIC=val_acc_overall   # or val_acc_worst

# # ---------- Outputs ----------
# STAMP="$(date +%Y%m%d_%H%M%S)"
# OUTDIR="results_made_mlp_batch_${STAMP}"
# mkdir -p "$OUTDIR"
# echo "Writing outputs to: $OUTDIR"

# # ---------- Helper ----------
# run() {
#   local name="$1"; shift
#   local csv="$OUTDIR/${name}.csv"
#   local log="$OUTDIR/${name}.log"
#   echo ">>> Running: $name"
#   echo "COMMAND: $*" | tee "$log"
#   "$@" --out-csv "$csv" 2>&1 | tee -a "$log"
#   echo "Saved: $csv"
#   echo "Log:   $log"
#   echo
# }

# # ---------- Main ----------
# for SEP in "${CLASS_SEP_LIST[@]}"; do
#   for SIGH in "${SIGMA_CORE_LIST[@]}"; do
#     TAG="sep${SEP}_sig${SIGH}"
#     IDCFG="{\"core_scale\": ${SEP}, \"sigma_core\": ${SIGH}}"
#     OODCFG="{\"core_scale\": ${SEP}, \"sigma_core\": ${SIGH}}"

#     # Low-data (100 epochs)
#     run "low_data_${TAG}_100ep" \
#     $PY $SCRIPT \
#       --device $DEVICE \
#       --seeds $SEEDS \
#       --d-list $DLIST \
#       --head-hidden-list $HIDDEN \
#       --head-layers-list $LAYERS \
#       --regimes $REGIMES \
#       --n-test $NTEST --epochs 5000 --lr $LR --batch-size $BS \
#       --val-from-id-frac $VAL_FRAC --select-metric $SELECT_METRIC \
#       --id-cfg-json  "$IDCFG" \
#       --ood-cfg-json "$OODCFG" \
#       --counts-by-group-train-json '{"(1,1)":60,"(-1,-1)":60,"(1,-1)":10,"(-1,1)":10}'

#     # Medium-data (100 epochs)
#     run "med_data_${TAG}_100ep" \
#     $PY $SCRIPT \
#       --device $DEVICE \
#       --seeds $SEEDS \
#       --d-list $DLIST \
#       --head-hidden-list $HIDDEN \
#       --head-layers-list $LAYERS \
#       --regimes $REGIMES \
#       --n-test $NTEST --epochs 5000 --lr $LR --batch-size $BS \
#       --val-from-id-frac $VAL_FRAC --select-metric $SELECT_METRIC \
#       --id-cfg-json  "$IDCFG" \
#       --ood-cfg-json "$OODCFG" \
#       --counts-by-group-train-json '{"(1,1)":600,"(-1,-1)":600,"(1,-1)":60,"(-1,1)":60}'

#     # High-data (100 epochs)
#     run "high_data_${TAG}_100ep" \
#     $PY $SCRIPT \
#       --device $DEVICE \
#       --seeds $SEEDS \
#       --d-list $DLIST \
#       --head-hidden-list $HIDDEN \
#       --head-layers-list $LAYERS \
#       --regimes $REGIMES \
#       --n-test $NTEST --epochs 5000 --lr $LR --batch-size $BS \
#       --val-from-id-frac $VAL_FRAC --select-metric $SELECT_METRIC \
#       --id-cfg-json  "$IDCFG" \
#       --ood-cfg-json "$OODCFG" \
#       --counts-by-group-train-json '{"(1,1)":1200,"(-1,-1)":1200,"(1,-1)":200,"(-1,1)":200}'

#   done
# done
