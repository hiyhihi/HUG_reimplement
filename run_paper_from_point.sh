#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/quynhptit/huyptit/AAAI26-HUG
DATA_ROOT=${DATA_ROOT:-$PROJECT_DIR/data/fashion-iq}
POINT_CHECKPOINT=${POINT_CHECKPOINT:-$PROJECT_DIR/checkpoints/point_dress_seed42/checkpoint_best.pth}
source "$PROJECT_DIR/ref/LAVIS/.venv/bin/activate"
cd "$PROJECT_DIR"

if [[ ! -f "$POINT_CHECKPOINT" ]]; then
  echo "Missing point checkpoint: $POINT_CHECKPOINT" >&2
  echo "Run ./run_point_baseline.sh first." >&2
  exit 1
fi

python -u train.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category dress \
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
  --output_dir checkpoints/paper_dress_seed42 \
  --seed 42 \
  --use_wandb

python -u eval.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category dress \
  --checkpoint checkpoints/paper_dress_seed42/checkpoint_best.pth \
  --distance_mode probabilistic \
  --output_file results/paper_dress_seed42_probabilistic.json

python -u eval.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category dress \
  --checkpoint checkpoints/paper_dress_seed42/checkpoint_best.pth \
  --distance_mode mean \
  --output_file results/paper_dress_seed42_mean.json
