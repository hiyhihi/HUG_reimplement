#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CATEGORY=${1:-dress}
SEED=${SEED:-42}
USE_WANDB=${USE_WANDB:-0}
DATA_ROOT=${DATA_ROOT:-$PROJECT_DIR/data/fashion-iq}
OUTPUT_DIR="checkpoints/point_${CATEGORY}_seed${SEED}"

case "$CATEGORY" in
  dress|shirt|toptee) ;;
  *)
    echo "Usage: $0 [dress|shirt|toptee]" >&2
    exit 2
    ;;
esac

source "$PROJECT_DIR/ref/LAVIS/.venv/bin/activate"
export WANDB_CONSOLE=${WANDB_CONSOLE:-off}
WANDB_ARGS=()
[[ "$USE_WANDB" == 1 ]] && WANDB_ARGS+=(--use_wandb)
cd "$PROJECT_DIR"

python -u train.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category "$CATEGORY" \
  --recipe point \
  --batch_size 32 \
  --num_epochs 10 \
  --warmup_epochs 0 \
  --lr 3e-5 \
  --eval_every 1 \
  --save_interval 0 \
  --output_dir "$OUTPUT_DIR" \
  --use_wandb

python -u eval.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category "$CATEGORY" \
  --checkpoint "$OUTPUT_DIR/checkpoint_best.pth" \
  --distance_mode mean \
  --output_file "results/point_${CATEGORY}_seed${SEED}.json"
