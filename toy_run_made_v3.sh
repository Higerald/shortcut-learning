#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
PY=python
SCRIPT=toy_lr_made.py   # <- best-checkpoint selection & 100 epochs
SEEDS=0,1,2,3,4
DLIST=16,64,256
HIDDEN=64,256
LAYERS=1,2
REGIMES=erm,reweight,downsample,upsample
NTEST=4000
LR=1e-4
BS=256
DEVICE=cpu                      # MLP/MADE are light; swap to cuda if you like

# Sweep class separation via core_scale (means at ±core_scale)
# CLASS_SEP_LIST=("0.25" "0.50" "1.00")
CLASS_SEP_LIST=("0.25" "0.50" "0.75" "1.00" "1.25")

# (Optional) core feature noise; keep default 0.15 if you only want to vary separation
# SIGMA_CORE_LIST=("0.15")
SIGMA_CORE_LIST=("0.5" "1.0" "1.5")

# Validation from ID test
VAL_FRAC=0.2
SELECT_METRIC=val_acc_overall   # or val_acc_worst

# ---------- Outputs ----------
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results_made_mlp_batch_${STAMP}"
mkdir -p "$OUTDIR"
echo "Writing outputs to: $OUTDIR"

# ---------- Helper ----------
run() {
  local name="$1"; shift
  local csv="$OUTDIR/${name}.csv"
  local log="$OUTDIR/${name}.log"
  echo ">>> Running: $name"
  echo "COMMAND: $*" | tee "$log"
  "$@" --out-csv "$csv" 2>&1 | tee -a "$log"
  echo "Saved: $csv"
  echo "Log:   $log"
  echo
}

# ---------- Main ----------
for SEP in "${CLASS_SEP_LIST[@]}"; do
  for SIGH in "${SIGMA_CORE_LIST[@]}"; do
    TAG="sep${SEP}_sig${SIGH}"
    IDCFG="{\"core_scale\": ${SEP}, \"sigma_core\": ${SIGH}}"
    OODCFG="{\"core_scale\": ${SEP}, \"sigma_core\": ${SIGH}}"

    # Low-data (100 epochs)
    run "low_data_${TAG}_100ep" \
    $PY $SCRIPT \
      --device $DEVICE \
      --seeds $SEEDS \
      --d-list $DLIST \
      --head-hidden-list $HIDDEN \
      --head-layers-list $LAYERS \
      --regimes $REGIMES \
      --n-test $NTEST --epochs 5000 --lr $LR --batch-size $BS \
      --val-from-id-frac $VAL_FRAC --select-metric $SELECT_METRIC \
      --id-cfg-json  "$IDCFG" \
      --ood-cfg-json "$OODCFG" \
      --counts-by-group-train-json '{"(1,1)":60,"(-1,-1)":60,"(1,-1)":10,"(-1,1)":10}'

    # Medium-data (100 epochs)
    run "med_data_${TAG}_100ep" \
    $PY $SCRIPT \
      --device $DEVICE \
      --seeds $SEEDS \
      --d-list $DLIST \
      --head-hidden-list $HIDDEN \
      --head-layers-list $LAYERS \
      --regimes $REGIMES \
      --n-test $NTEST --epochs 5000 --lr $LR --batch-size $BS \
      --val-from-id-frac $VAL_FRAC --select-metric $SELECT_METRIC \
      --id-cfg-json  "$IDCFG" \
      --ood-cfg-json "$OODCFG" \
      --counts-by-group-train-json '{"(1,1)":600,"(-1,-1)":600,"(1,-1)":60,"(-1,1)":60}'

    # High-data (100 epochs)
    run "high_data_${TAG}_100ep" \
    $PY $SCRIPT \
      --device $DEVICE \
      --seeds $SEEDS \
      --d-list $DLIST \
      --head-hidden-list $HIDDEN \
      --head-layers-list $LAYERS \
      --regimes $REGIMES \
      --n-test $NTEST --epochs 5000 --lr $LR --batch-size $BS \
      --val-from-id-frac $VAL_FRAC --select-metric $SELECT_METRIC \
      --id-cfg-json  "$IDCFG" \
      --ood-cfg-json "$OODCFG" \
      --counts-by-group-train-json '{"(1,1)":1200,"(-1,-1)":1200,"(1,-1)":200,"(-1,1)":200}'

  done
done
