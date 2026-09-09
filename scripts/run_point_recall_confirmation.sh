#!/usr/bin/env bash
set -euo pipefail

# Confirm whether the current point_robust gain repeats on an independent Dress
# seed before opening Shirt/Toptee or sweeping dropout/lambda.
#
# Run one seed at a time:
#   CONFIRM_SEED=202 RUN_ROBUSTNESS=0 bash scripts/run_point_recall_confirmation.sh
#   CONFIRM_SEED=3407 RUN_ROBUSTNESS=0 bash scripts/run_point_recall_confirmation.sh
#
# Only evaluate robustness after the clean-recall gate passes:
#   CONFIRM_SEED=202 RUN_ROBUSTNESS=1 bash scripts/run_point_recall_confirmation.sh

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$ROOT_DIR/data/fashion-iq"}
VENV=${VENV:-"$ROOT_DIR/ref/LAVIS/.venv/bin/activate"}
RESULT_ROOT=${RESULT_ROOT:-"$ROOT_DIR/results/point_recall_confirmation_v2"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$ROOT_DIR/checkpoints/point_recall_confirmation_v2"}
CONFIRM_SEED=${CONFIRM_SEED:-202}
RUN_ROBUSTNESS=${RUN_ROBUSTNESS:-0}
NUM_WORKERS=${NUM_WORKERS:-4}

[[ "$CONFIRM_SEED" =~ ^[0-9]+$ ]] || {
  echo "CONFIRM_SEED must be a non-negative integer" >&2
  exit 2
}
[[ "$RUN_ROBUSTNESS" == 0 || "$RUN_ROBUSTNESS" == 1 ]] || {
  echo "RUN_ROBUSTNESS must be 0 or 1" >&2
  exit 2
}

[[ -f "$VENV" ]] && source "$VENV"
cd "$ROOT_DIR"

echo "Stage 1/3: original Point baseline, Dress seed $CONFIRM_SEED"
DATA_ROOT="$DATA_ROOT" \
RESULT_ROOT="$RESULT_ROOT" \
CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
SEEDS="$CONFIRM_SEED" \
CATEGORIES=dress \
MODELS=point \
NUM_WORKERS="$NUM_WORKERS" \
"$ROOT_DIR/scripts/run_supervisor_experiments.sh" categories

if [[ "$RUN_ROBUSTNESS" == 1 ]]; then
  echo "Robustness evaluation: original Point, Dress seed $CONFIRM_SEED"
  DATA_ROOT="$DATA_ROOT" \
  RESULT_ROOT="$RESULT_ROOT" \
  CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
  SEEDS="$CONFIRM_SEED" \
  CATEGORIES=dress \
  MODELS=point \
  NUM_WORKERS="$NUM_WORKERS" \
  "$ROOT_DIR/scripts/run_supervisor_experiments.sh" robustness
fi

POINT_CHECKPOINT="$CHECKPOINT_ROOT/point/dress/seed$CONFIRM_SEED/checkpoint_best.pth"
[[ -f "$POINT_CHECKPOINT" ]] || {
  echo "Missing Point checkpoint: $POINT_CHECKPOINT" >&2
  exit 1
}

echo "Stage 2/3: Point continued-clean control, Dress seed $CONFIRM_SEED"
DATA_ROOT="$DATA_ROOT" \
RESULT_ROOT="$RESULT_ROOT" \
CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
CATEGORY=dress \
SEED="$CONFIRM_SEED" \
POINT_CHECKPOINT="$POINT_CHECKPOINT" \
BATCH_SIZE=32 \
EPOCHS=8 \
LR=3e-6 \
PATIENCE=2 \
NUM_WORKERS="$NUM_WORKERS" \
RUN_ROBUSTNESS="$RUN_ROBUSTNESS" \
"$ROOT_DIR/scripts/run_point_continued_control.sh"

echo "Stage 3/3: Point Robust with the unchanged objective, Dress seed $CONFIRM_SEED"
DATA_ROOT="$DATA_ROOT" \
RESULT_ROOT="$RESULT_ROOT" \
CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
CATEGORY=dress \
SEED="$CONFIRM_SEED" \
POINT_CHECKPOINT="$POINT_CHECKPOINT" \
BATCH_SIZE=32 \
EPOCHS=8 \
LR=3e-6 \
PATIENCE=2 \
TEXT_DROPOUT=0.10 \
LAMBDA_ROBUST=0.5 \
LAMBDA_CONSISTENCY=0.5 \
NUM_WORKERS="$NUM_WORKERS" \
RUN_ROBUSTNESS="$RUN_ROBUSTNESS" \
"$ROOT_DIR/scripts/run_point_robust_recall.sh"

echo "Completed Dress confirmation seed $CONFIRM_SEED in $RESULT_ROOT"
