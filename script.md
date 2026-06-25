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

python train.py \
    --dataset fashion-iq \
    --data_root /mnt/data/users/quynhptit/huyptit/AAAI26-HUG/data/fashion-iq \
    --category dress \
    --batch_size 32 \
    --num_epochs 30 \
    --lr 3e-5 \
    --lambda_fc 0.5 \
    --lambda_cord 0.1 \
    --output_dir ./checkpoints/fashion_iq \
    --use_wandb

python eval.py \
    --dataset fashion-iq \
    --data_root /mnt/data/users/quynhptit/huyptit/AAAI26-HUG/data/fashion-iq \
    --category dress \
    --checkpoint ./checkpoints/fashion_iq/checkpoint_final.pth \
    --batch_size 64 \
    --output_file ./results/fashion_iq_dress_results.json


Evaluation Results:
==================================================
recall@1: 10.91
recall@5: 25.93
recall@10: 34.46
recall@50: 59.44
mrr: 0.19
median_rank: 30
==================================================
