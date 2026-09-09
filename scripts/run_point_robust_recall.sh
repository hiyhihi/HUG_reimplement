#!/usr/bin/env bash
set -euo pipefail

# Recall-oriented Point ablation: clean query + mild token-dropout query.
# It uses a new output root and never overwrites the completed Point baseline.
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$ROOT_DIR/data/fashion-iq"}
VENV=${VENV:-"$ROOT_DIR/ref/LAVIS/.venv/bin/activate"}
RESULT_ROOT=${RESULT_ROOT:-"$ROOT_DIR/results/point_robust_recall_v1"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$ROOT_DIR/checkpoints/point_robust_recall_v1"}
CATEGORY=${CATEGORY:-dress}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-8}
LR=${LR:-3e-6}
TEXT_DROPOUT=${TEXT_DROPOUT:-0.10}
LAMBDA_ROBUST=${LAMBDA_ROBUST:-0.5}
LAMBDA_CONSISTENCY=${LAMBDA_CONSISTENCY:-0.5}
PATIENCE=${PATIENCE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
RUN_ROBUSTNESS=${RUN_ROBUSTNESS:-1}
POINT_CHECKPOINT=${POINT_CHECKPOINT:-"$ROOT_DIR/checkpoints/supervisor_protocol_v2/point/$CATEGORY/seed$SEED/checkpoint_best.pth"}

[[ -f "$VENV" ]] && source "$VENV"
cd "$ROOT_DIR"

[[ -f "$POINT_CHECKPOINT" ]] || {
  echo "Missing Point checkpoint: $POINT_CHECKPOINT" >&2
  exit 1
}
[[ "$RUN_ROBUSTNESS" == 0 || "$RUN_ROBUSTNESS" == 1 ]] || {
  echo "RUN_ROBUSTNESS must be 0 or 1" >&2
  exit 2
}
(( BATCH_SIZE >= 2 )) || { echo "BATCH_SIZE must be >= 2" >&2; exit 2; }

RUN_DIR="$CHECKPOINT_ROOT/point_robust/$CATEGORY/seed$SEED"
LOG_FILE="$RESULT_ROOT/logs/point_robust_${CATEGORY}_seed${SEED}.log"
CLEAN_FILE="$RESULT_ROOT/clean/point_robust_${CATEGORY}_seed${SEED}.json"
ROBUST_DIR="$RESULT_ROOT/robustness"
ROBUST_MARKER="$ROBUST_DIR/.point_robust_${CATEGORY}_seed${SEED}.complete"
mkdir -p "$RUN_DIR" "$RESULT_ROOT/logs" "$RESULT_ROOT/clean" "$ROBUST_DIR"

if [[ -f "$RUN_DIR/checkpoint_final.pth" ]]; then
  echo "SKIP completed training: $RUN_DIR/checkpoint_final.pth"
else
  start_args=(--init_checkpoint "$POINT_CHECKPOINT")
  if [[ -f "$RUN_DIR/checkpoint_last.pth" ]]; then
    start_args=(--resume_checkpoint "$RUN_DIR/checkpoint_last.pth")
    echo "RESUME: $RUN_DIR/checkpoint_last.pth"
  elif [[ -f "$RUN_DIR/checkpoint_best.pth" ]]; then
    echo "Incomplete run has checkpoint_best but no checkpoint_last: $RUN_DIR" >&2
    exit 1
  fi

  python -u train.py \
    --dataset fashion-iq --data_root "$DATA_ROOT" --category "$CATEGORY" --seed "$SEED" \
    --recipe point_robust --batch_size "$BATCH_SIZE" --num_epochs "$EPOCHS" \
    --warmup_epochs 0 --lr "$LR" --loss_lr_multiplier 1 \
    --robust_text_dropout_rate "$TEXT_DROPOUT" \
    --lambda_robust_retrieval "$LAMBDA_ROBUST" \
    --lambda_robust_consistency "$LAMBDA_CONSISTENCY" \
    --eval_every 1 --early_stopping_patience "$PATIENCE" --early_stopping_min_delta 0 \
    --save_interval 0 --log_instrumentation --instrumentation_every 0 \
    --num_workers "$NUM_WORKERS" --tqdm_mininterval 1 \
    --output_dir "$RUN_DIR" "${start_args[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

BEST="$RUN_DIR/checkpoint_best.pth"
[[ -f "$BEST" ]] || { echo "Missing best checkpoint after training: $BEST" >&2; exit 1; }

if [[ -f "$CLEAN_FILE" ]]; then
  echo "SKIP clean evaluation: $CLEAN_FILE"
else
  python -u eval.py \
    --dataset fashion-iq --data_root "$DATA_ROOT" --split val --category "$CATEGORY" \
    --checkpoint "$BEST" --batch_size 32 --num_workers "$NUM_WORKERS" \
    --distance_mode mean --output_file "$CLEAN_FILE"
fi

if [[ "$RUN_ROBUSTNESS" == 1 && -f "$ROBUST_MARKER" ]]; then
  echo "SKIP completed robustness sweep: $ROBUST_MARKER"
elif [[ "$RUN_ROBUSTNESS" == 1 ]]; then
  python -u eval/robustness.py sweep \
    --dataset fashion-iq --data_root "$DATA_ROOT" --split val --category "$CATEGORY" \
    --seed "$SEED" --checkpoint "$BEST" --model_id point_robust \
    --batch_size 32 --num_workers "$NUM_WORKERS" --skip_uncertainty \
    --distance_modes mean --severities 1 2 3 4 --output_dir "$ROBUST_DIR"
  touch "$ROBUST_MARKER"
fi

echo "Completed Point-robust pilot in $RESULT_ROOT"
