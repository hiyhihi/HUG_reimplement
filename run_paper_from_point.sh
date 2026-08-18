#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CATEGORY=${1:-dress}
SEED=${SEED:-42}
USE_WANDB=${USE_WANDB:-0}
DATA_ROOT=${DATA_ROOT:-$PROJECT_DIR/data/fashion-iq}
POINT_CHECKPOINT=${POINT_CHECKPOINT:-$PROJECT_DIR/checkpoints/point_${CATEGORY}_seed${SEED}/checkpoint_best.pth}
OUTPUT_DIR="checkpoints/paper_${CATEGORY}_seed${SEED}"

case "$CATEGORY" in
  dress|shirt|toptee) ;;
  *)
    echo "Usage: $0 [dress|shirt|toptee]" >&2
    exit 2
    ;;
esac

export WANDB_CONSOLE=${WANDB_CONSOLE:-off}
WANDB_ARGS=()
[[ "$USE_WANDB" == 1 ]] && WANDB_ARGS+=(--use_wandb)
source "$PROJECT_DIR/ref/LAVIS/.venv/bin/activate"
cd "$PROJECT_DIR"

if [[ ! -f "$POINT_CHECKPOINT" ]]; then
  echo "Missing point checkpoint: $POINT_CHECKPOINT" >&2
  echo "Run ./run_point_baseline.sh $CATEGORY first." >&2
  exit 1
fi

python -u train.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category "$CATEGORY" \
  --recipe paper \
  --init_checkpoint "$POINT_CHECKPOINT" \
  --freeze_backbone \
  --batch_size 32 \
  --num_epochs 20 \
  --warmup_epochs 0 \
  --lr 3e-5 \
  --loss_lr_multiplier 10 \
  --lambda_fc 0.5 \
  --lambda_cord 0.1 \
  --eval_every 1 \
  --save_interval 0 \
  --output_dir "$OUTPUT_DIR" \
  --use_wandb

python -u eval.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category "$CATEGORY" \
  --checkpoint "$OUTPUT_DIR/checkpoint_best.pth" \
  --distance_mode probabilistic \
  --output_file "results/paper_${CATEGORY}_seed${SEED}_probabilistic.json"

python -u eval.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category "$CATEGORY" \
  --checkpoint "$OUTPUT_DIR/checkpoint_best.pth" \
  --distance_mode mean \
  --output_file "results/paper_${CATEGORY}_seed${SEED}_mean.json"
