#!/bin/bash

# 1. Kích hoạt môi trường ảo chắc chắn 100%
source /mnt/data/users/quynhptit/huyptit/AAAI26-HUG/ref/LAVIS/.venv/bin/activate

# 2. Chạy python với cờ "-u" (UNBUFFERED - Cực kỳ quan trọng)
python -u train.py \
    --dataset fashion-iq \
    --data_root /mnt/data/users/quynhptit/huyptit/AAAI26-HUG/data/fashion-iq \
    --category dress \
    --batch_size 32 \
    --num_epochs 6 \
    --lr 3e-5 \
    --lambda_fc 0.5 \
    --lambda_cord 0.1 \
    --output_dir ./checkpoints/fashion_iq \
    --use_wandb
