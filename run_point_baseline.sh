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
  --recipe point \
  --batch_size 32 \
  --num_epochs 10 \
  --warmup_epochs 0 \
  --lr 3e-5 \
  --eval_every 1 \
  --output_dir checkpoints/point_dress_seed42 \
  --seed 42 \
  --use_wandb

python -u eval.py \
  --dataset fashion-iq \
  --data_root "$DATA_ROOT" \
  --category dress \
  --checkpoint checkpoints/point_dress_seed42/checkpoint_best.pth \
  --distance_mode mean \
  --output_file results/point_dress_seed42.json
