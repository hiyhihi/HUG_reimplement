#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/quynhptit/huyptit/AAAI26-HUG
DATA_ROOT=${DATA_ROOT:-$PROJECT_DIR/data/fashion-iq}
source "$PROJECT_DIR/ref/LAVIS/.venv/bin/activate"
cd "$PROJECT_DIR"

python -u train.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category dress \
  --recipe legacy \
  --batch_size 32 \
  --num_epochs 6 \
  --warmup_epochs 0 \
  --lr 3e-5 \
  --loss_lr_multiplier 100 \
  --lambda_fc 0.5 \
  --lambda_cord 0.1 \
  --eval_every 0 \
  --output_dir checkpoints/legacy_dress_seed42 \
  --seed 42 \
  --use_wandb

python -u eval.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category dress \
  --checkpoint checkpoints/legacy_dress_seed42/checkpoint_final.pth \
  --distance_mode probabilistic \
  --output_file results/legacy_dress_seed42.json
