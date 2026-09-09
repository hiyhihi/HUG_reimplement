#!/usr/bin/env bash
set -euo pipefail

# Two small, isolated experiments for the HUG HC collapse. They never write to
# supervisor_protocol_v2, so completed baseline artifacts remain untouched.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$ROOT_DIR/data/fashion-iq"}
VENV=${VENV:-"$ROOT_DIR/ref/LAVIS/.venv/bin/activate"}
RESULT_ROOT=${RESULT_ROOT:-"$ROOT_DIR/results/supervisor_diagnostics_v2"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$ROOT_DIR/checkpoints/supervisor_diagnostics_v2"}
SEED=${SEED:-42}
CATEGORY=${CATEGORY:-dress}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-30}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-2}
PATIENCE=${PATIENCE:-2}
LR=${LR:-3e-5}
NUM_WORKERS=${NUM_WORKERS:-4}
FORCE=${FORCE:-0}

[[ -f "$VENV" ]] && source "$VENV"
cd "$ROOT_DIR"
mkdir -p "$RESULT_ROOT/clean" "$RESULT_ROOT/logs" "$CHECKPOINT_ROOT"

[[ "$FORCE" == 0 || "$FORCE" == 1 ]] || { echo 'FORCE must be 0 or 1' >&2; exit 2; }
(( BATCH_SIZE >= 2 )) || { echo 'BATCH_SIZE must be >= 2' >&2; exit 2; }
(( PATIENCE >= 1 )) || { echo 'PATIENCE must be >= 1' >&2; exit 2; }

run_one() {
  local name=$1
  local flag=$2
  local checkpoint_dir="$CHECKPOINT_ROOT/$name/$CATEGORY/seed$SEED"
  local log_file="$RESULT_ROOT/logs/${name}_${CATEGORY}_seed${SEED}.log"
  local resume_args=()

  if [[ "$FORCE" != 1 && -f "$checkpoint_dir/checkpoint_final.pth" ]]; then
    echo "SKIP completed diagnostic: $checkpoint_dir/checkpoint_final.pth"
  else
    mkdir -p "$checkpoint_dir"
    if [[ "$FORCE" != 1 && -f "$checkpoint_dir/checkpoint_last.pth" ]]; then
      resume_args=(--resume_checkpoint "$checkpoint_dir/checkpoint_last.pth")
      echo "RESUME diagnostic: $checkpoint_dir/checkpoint_last.pth"
    elif [[ "$FORCE" != 1 && -f "$checkpoint_dir/checkpoint_best.pth" ]]; then
      echo "ERROR: incomplete run has checkpoint_best but no resumable checkpoint_last: $checkpoint_dir" >&2
      exit 1
    fi
    python -u train.py \
      --dataset fashion-iq --data_root "$DATA_ROOT" --category "$CATEGORY" --seed "$SEED" \
      --batch_size "$BATCH_SIZE" --num_epochs "$EPOCHS" --warmup_epochs "$WARMUP_EPOCHS" \
      --lr "$LR" --recipe paper --lambda_fc 0 --lambda_cord 0 --loss_lr_multiplier 1 \
      --eval_every 1 --early_stopping_patience "$PATIENCE" --early_stopping_min_delta 0 \
      --log_instrumentation --instrumentation_every 0 \
      --save_interval 0 --num_workers "$NUM_WORKERS" --tqdm_mininterval 1 \
      --output_dir "$checkpoint_dir" "$flag" "${resume_args[@]}" 2>&1 | tee -a "$log_file"
  fi

  [[ -f "$checkpoint_dir/checkpoint_best.pth" ]] || {
    echo "ERROR: missing checkpoint_best after training: $checkpoint_dir" >&2
    exit 1
  }

  for mode in mean probabilistic; do
    local output="$RESULT_ROOT/clean/${name}_${CATEGORY}_seed${SEED}_${mode}.json"
    if [[ "$FORCE" != 1 && -f "$output" ]]; then
      echo "SKIP eval: $output"
      continue
    fi
    python -u eval.py --dataset fashion-iq --data_root "$DATA_ROOT" --split val \
      --category "$CATEGORY" --checkpoint "$checkpoint_dir/checkpoint_best.pth" \
      --batch_size 32 --num_workers "$NUM_WORKERS" --distance_mode "$mode" --output_file "$output"
  done
}

# A: Does the sigmoid HC work when its distance ignores uncertainty entirely?
run_one hug_hc_mean_distance --hc_mean_distance

# B: Does HC fail because each query's negatives are summed rather than averaged?
run_one hug_hc_average_negatives --hc_average_negatives

echo "Completed HC diagnostics in $RESULT_ROOT"
