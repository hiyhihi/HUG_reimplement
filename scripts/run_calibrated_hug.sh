#!/usr/bin/env bash
# Train one RC-HUG calibration stage. Usage: ./scripts/run_calibrated_hug.sh {u1|u2|u3} [dress|shirt|toptee]
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE=${1:?"Usage: $0 {u1|u2|u3} [dress|shirt|toptee]"}
CATEGORY=${2:-dress}
SEED=${SEED:-42}
DATA_ROOT=${DATA_ROOT:-$ROOT_DIR/data/fashion-iq}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
USE_WANDB=${USE_WANDB:-1}
TEACHER_DEVICE=${TEACHER_DEVICE:-cuda}
VENV=${VENV:-$ROOT_DIR/ref/LAVIS/.venv/bin/activate}

case "$CATEGORY" in dress|shirt|toptee) ;; *) echo "Category must be dress, shirt, or toptee" >&2; exit 2;; esac
case "$STAGE" in u1|u2|u2_heads|u3) ;; *) echo "Stage must be u1, u2, u2_heads, or u3" >&2; exit 2;; esac
[[ -f "$VENV" ]] && source "$VENV"
cd "$ROOT_DIR"

POINT_CHECKPOINT=${POINT_CHECKPOINT:-$ROOT_DIR/checkpoints/point_${CATEGORY}_seed${SEED}/checkpoint_best.pth}
PAPER_CHECKPOINT=${PAPER_CHECKPOINT:-$ROOT_DIR/checkpoints/paper_${CATEGORY}_seed${SEED}/checkpoint_best.pth}
# U1 optimizes calibration, not clean Recall; its final epoch is the default initializer for U2.
U1_CHECKPOINT=${U1_CHECKPOINT:-$ROOT_DIR/checkpoints/u1_${CATEGORY}_seed${SEED}/checkpoint_final.pth}
U2_CHECKPOINT=${U2_CHECKPOINT:-$ROOT_DIR/checkpoints/u2_${CATEGORY}_seed${SEED}/checkpoint_best.pth}
U2_HEADS_CHECKPOINT=${U2_HEADS_CHECKPOINT:-$ROOT_DIR/checkpoints/u2_heads_${CATEGORY}_seed${SEED}/checkpoint_best.pth}
OUTPUT_DIR="checkpoints/${STAGE}_${CATEGORY}_seed${SEED}"
WANDB_ARGS=()
[[ "$USE_WANDB" == 1 ]] && WANDB_ARGS+=(--use_wandb)

COMMON=(
  --dataset fashion-iq --data_root "$DATA_ROOT" --category "$CATEGORY" --recipe paper
  --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" --num_epochs "$EPOCHS"
  --warmup_epochs 0 --lr 3e-5 --loss_lr_multiplier 10 --lambda_fc 0.5 --lambda_cord 0.1
  --eval_every 1 --output_dir "$OUTPUT_DIR" --seed "$SEED"
)

case "$STAGE" in
  u1)
    INIT_CHECKPOINT="$PAPER_CHECKPOINT"
    EXTRA=(--freeze_backbone --lambda_monotonic 0.1 --monotonic_margin 0.005 --monotonic_low_severity 0.10 --monotonic_high_severity 0.30 --monotonic_modality alternate)
    ;;
  u2|u2_heads)
    INIT_CHECKPOINT="$U1_CHECKPOINT"
    EXTRA=(--lambda_monotonic 0.1 --monotonic_margin 0.005 --monotonic_low_severity 0.10 --monotonic_high_severity 0.30 --monotonic_modality alternate --modality_dropout_prob 0.15 --modality_dropout_text_rate 0.30)
    if [[ "$STAGE" == u2_heads ]]; then
      # Heads-only ablation: preserve the strong mean representation and fit uncertainty heads only.
      EXTRA+=(--freeze_backbone)
    fi
    ;;
  u3)
    # U2 full-finetune was constrained to a reduced physical batch after OOM
    # and lost 16.8 R@10. Start the KD test from the validated heads-only U2
    # checkpoint so it isolates whether KD preserves clean retrieval.
    INIT_CHECKPOINT="$U2_HEADS_CHECKPOINT"
    [[ -f "$POINT_CHECKPOINT" ]] || { echo "Missing Point teacher: $POINT_CHECKPOINT" >&2; exit 1; }
    # U1/U2-head already calibrated the uncertainty heads. Repeating its paired
    # loss adds two full student forwards per batch without isolating KD.
    # Run a short, GPU-teacher pilot first; set U3_EPOCHS/U3_BATCH_SIZE to tune.
    COMMON[9]=${U3_BATCH_SIZE:-4}
    COMMON[13]=${U3_EPOCHS:-3}
    EXTRA=(--modality_dropout_prob 0.10 --modality_dropout_text_rate 0.30 --lambda_kd 1.0 --teacher_checkpoint "$POINT_CHECKPOINT" --teacher_device "$TEACHER_DEVICE" --kd_temperature 0.07)
    ;;
esac

[[ -f "$INIT_CHECKPOINT" ]] || { echo "Missing initialization checkpoint: $INIT_CHECKPOINT" >&2; exit 1; }
echo "Running $STAGE for $CATEGORY (seed=$SEED); init=$INIT_CHECKPOINT; output=$OUTPUT_DIR"
python -u train.py "${COMMON[@]}" --init_checkpoint "$INIT_CHECKPOINT" "${EXTRA[@]}" "${WANDB_ARGS[@]}"

# U1's objective is calibration; clean-R@K checkpoint selection is not a calibration selector.
SELECTED_CHECKPOINT="$OUTPUT_DIR/checkpoint_best.pth"
[[ "$STAGE" == u1 ]] && SELECTED_CHECKPOINT="$OUTPUT_DIR/checkpoint_final.pth"
for DISTANCE_MODE in probabilistic mean; do
  python -u eval.py --dataset fashion-iq --data_root "$DATA_ROOT" --category "$CATEGORY" \
    --checkpoint "$SELECTED_CHECKPOINT" --distance_mode "$DISTANCE_MODE" \
    --output_file "results/${STAGE}_${CATEGORY}_seed${SEED}_${DISTANCE_MODE}.json"
done

echo "Completed $STAGE; selected checkpoint: $SELECTED_CHECKPOINT"
echo "Next: evaluate it with scripts/run_person2_timeline.sh custom."
