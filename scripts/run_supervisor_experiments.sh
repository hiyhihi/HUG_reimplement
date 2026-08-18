#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$ROOT_DIR/data/fashion-iq"}
VENV=${VENV:-"$ROOT_DIR/ref/LAVIS/.venv/bin/activate"}
RESULT_ROOT=${RESULT_ROOT:-"$ROOT_DIR/results/supervisor_protocol_v2"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$ROOT_DIR/checkpoints/supervisor_protocol_v2"}
SEEDS_CSV=${SEEDS:-42,7,123}
CATEGORIES_CSV=${CATEGORIES:-dress,shirt,toptee}
MODELS_CSV=${MODELS:-point,point_matched,hug_e2e,hug_frozen_point}
POINT_BATCH_SIZE=${POINT_BATCH_SIZE:-32}
POINT_EPOCHS=${POINT_EPOCHS:-10}
POINT_WARMUP_EPOCHS=${POINT_WARMUP_EPOCHS:-0}
E2E_BATCH_SIZE=${E2E_BATCH_SIZE:-8}
E2E_EPOCHS=${E2E_EPOCHS:-30}
E2E_WARMUP_EPOCHS=${E2E_WARMUP_EPOCHS:-2}
FROZEN_BATCH_SIZE=${FROZEN_BATCH_SIZE:-32}
FROZEN_EPOCHS=${FROZEN_EPOCHS:-20}
FROZEN_WARMUP_EPOCHS=${FROZEN_WARMUP_EPOCHS:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-3e-5}
FORCE=${FORCE:-0}
USE_WANDB=${USE_WANDB:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a CATEGORIES <<< "$CATEGORIES_CSV"
IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
export WANDB_CONSOLE=${WANDB_CONSOLE:-off}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
[[ -f "$VENV" ]] && source "$VENV"
cd "$ROOT_DIR"
mkdir -p "$RESULT_ROOT/clean" "$RESULT_ROOT/modality" "$RESULT_ROOT/robustness" "$RESULT_ROOT/tables" "$CHECKPOINT_ROOT"

for model in "${MODELS[@]}"; do
  case "$model" in
    point|point_matched|hug_e2e|hug_frozen_point) ;;
    *) echo "Unknown MODELS entry: $model" >&2; exit 2 ;;
  esac
done
for batch in "$POINT_BATCH_SIZE" "$E2E_BATCH_SIZE" "$FROZEN_BATCH_SIZE"; do
  (( batch >= 2 )) || { echo "Every contrastive batch size must be >= 2" >&2; exit 2; }
done
[[ "$USE_WANDB" == 0 || "$USE_WANDB" == 1 ]] || { echo "USE_WANDB must be 0 or 1" >&2; exit 2; }
[[ "$FORCE" == 0 || "$FORCE" == 1 ]] || { echo "FORCE must be 0 or 1" >&2; exit 2; }


checkpoint_dir() {
  printf '%s/%s/%s/seed%s' "$CHECKPOINT_ROOT" "$1" "$2" "$3"
}

train_one() {
  local model=$1 category=$2 seed=$3
  local output point batch epochs warmup
  local args=() extra=() init_args=()
  output=$(checkpoint_dir "$model" "$category" "$seed")
  if [[ "$FORCE" != 1 && -f "$output/checkpoint_final.pth" ]]; then
    echo "SKIP completed train: $output/checkpoint_final.pth"
    return
  fi
  case "$model" in
    point)
      batch=$POINT_BATCH_SIZE
      epochs=$POINT_EPOCHS
      warmup=$POINT_WARMUP_EPOCHS
      extra+=(--recipe point)
      ;;
    point_matched)
      batch=$E2E_BATCH_SIZE
      epochs=$E2E_EPOCHS
      warmup=$E2E_WARMUP_EPOCHS
      extra+=(--recipe point)
      ;;
    hug_e2e)
      batch=$E2E_BATCH_SIZE
      epochs=$E2E_EPOCHS
      warmup=$E2E_WARMUP_EPOCHS
      extra+=(--recipe paper --lambda_fc 0.5 --lambda_cord 0.1 --loss_lr_multiplier 1)
      ;;
    hug_frozen_point)
      batch=$FROZEN_BATCH_SIZE
      epochs=$FROZEN_EPOCHS
      warmup=$FROZEN_WARMUP_EPOCHS
      point=$(checkpoint_dir point "$category" "$seed")/checkpoint_best.pth
      [[ -f "$point" ]] || { echo "Missing Point initializer: $point" >&2; return 1; }
      init_args+=(--init_checkpoint "$point")
      extra+=(--recipe paper --freeze_backbone
              --lambda_fc 0.5 --lambda_cord 0.1 --loss_lr_multiplier 10)
      ;;
    *)
      echo "Unknown model: $model" >&2
      return 2
      ;;
  esac
  args=(
    --dataset fashion-iq --data_root "$DATA_ROOT" --category "$category"
    --batch_size "$batch" --num_epochs "$epochs"
    --warmup_epochs "$warmup" --lr "$LR" --eval_every 1
    --output_dir "$output" --seed "$seed" --save_interval "$SAVE_INTERVAL"
    --num_workers "$NUM_WORKERS" --tqdm_mininterval 1
  )
  if [[ "$USE_WANDB" == 1 ]]; then
    args+=(--use_wandb)
  fi
  if [[ "$FORCE" != 1 && -f "$output/checkpoint_last.pth" ]]; then
    echo "RESUME train: $output/checkpoint_last.pth"
    args+=(--resume_checkpoint "$output/checkpoint_last.pth")
  elif [[ "$FORCE" != 1 && -f "$output/checkpoint_best.pth" ]]; then
    echo "Incomplete legacy checkpoint cannot be resumed safely: $output" >&2
    return 1
  else
    args+=("${init_args[@]}")
  fi
  mkdir -p "$output"
  echo "TRAIN model=$model category=$category seed=$seed batch=$batch epochs=$epochs warmup=$warmup output=$output"
  python -u train.py "${args[@]}" "${extra[@]}" 2>&1 | tee -a "$output/train.log"
}

eval_clean() {
  local model=$1 category=$2 seed=$3
  local checkpoint mode output
  checkpoint=$(checkpoint_dir "$model" "$category" "$seed")/checkpoint_best.pth
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; return 1; }
  local modes=(mean)
  [[ "$model" != point && "$model" != point_matched ]] && modes+=(probabilistic)
  for mode in "${modes[@]}"; do
    output="$RESULT_ROOT/clean/${model}_${category}_seed${seed}_${mode}.json"
    if [[ "$FORCE" != 1 && -f "$output" ]]; then
      echo "SKIP eval: $output"
      continue
    fi
    python -u eval.py --dataset fashion-iq --data_root "$DATA_ROOT" --split val \
      --category "$category" --checkpoint "$checkpoint" --batch_size "$EVAL_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" --distance_mode "$mode" --output_file "$output"
  done
}

run_train_set() {
  local categories=("$@")
  for category in "${categories[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for model in "${MODELS[@]}"; do
        train_one "$model" "$category" "$seed"
        eval_clean "$model" "$category" "$seed"
      done
    done
  done
}

run_modality() {
  local models=("${MODELS[@]}")
  for category in "${CATEGORIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for model in "${models[@]}"; do
        local checkpoint output extra modes
        checkpoint=$(checkpoint_dir "$model" "$category" "$seed")/checkpoint_best.pth
        [[ -f "$checkpoint" ]] || { echo "SKIP missing: $checkpoint"; continue; }
        output="$RESULT_ROOT/modality/${model}_${category}_seed${seed}.json"
        [[ "$FORCE" != 1 && -f "$output" ]] && { echo "SKIP modality: $output"; continue; }
        extra=(); modes=(mean)
        if [[ "$model" == point || "$model" == point_matched ]]; then extra+=(--skip_uncertainty); else modes+=(probabilistic); fi
        python -u eval/modality_reliance.py --dataset fashion-iq --data_root "$DATA_ROOT" \
          --category "$category" --checkpoint "$checkpoint" --model_id "$model" \
          --seed "$seed" --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
          --distance_modes "${modes[@]}" --output_file "$output" "${extra[@]}"
      done
    done
  done
}

run_robustness() {
  local models=("${MODELS[@]}")
  for category in "${CATEGORIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for model in "${models[@]}"; do
        local checkpoint output extra modes
        checkpoint=$(checkpoint_dir "$model" "$category" "$seed")/checkpoint_best.pth
        [[ -f "$checkpoint" ]] || { echo "SKIP missing: $checkpoint"; continue; }
        output="$RESULT_ROOT/robustness/$model/$category/seed$seed"
        if [[ "$FORCE" != 1 && -f "$output/.complete" ]]; then
          echo "SKIP robustness complete: $output"
          continue
        fi
        extra=(); modes=(mean)
        if [[ "$model" == point || "$model" == point_matched ]]; then
          extra+=(--skip_uncertainty)
        else
          modes+=(probabilistic)
          extra+=(--include_mismatch)
        fi
        python -u eval/robustness.py sweep --dataset fashion-iq --data_root "$DATA_ROOT" \
          --split val --category "$category" --checkpoint "$checkpoint" --model_id "$model" \
          --seed "$seed" --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
          --distance_modes "${modes[@]}" --output_dir "$output" "${extra[@]}"
        python eval/robustness.py aggregate \
          --input_glob "$output/*.json" \
          --output_file "$output/summary.json"
        touch "$output/.complete"
      done
    done
  done
}

case "${1:-help}" in
  preflight)
    python -m py_compile train.py eval.py eval/robustness.py eval/modality_reliance.py eval/summarize_supervisor.py
    [[ -f "$ROOT_DIR/2601.11393v2 (1).pdf" ]]
    for category in dress shirt toptee; do
      [[ -f "$DATA_ROOT/captions/cap.$category.train.json" ]]
      [[ -f "$DATA_ROOT/captions/cap.$category.val.json" ]]
    done
    echo "Preflight passed."
    echo "Models=$MODELS_CSV Seeds=$SEEDS_CSV Categories=$CATEGORIES_CSV"
    echo "Point: batch=$POINT_BATCH_SIZE epochs=$POINT_EPOCHS warmup=$POINT_WARMUP_EPOCHS"
    echo "E2E/matched: batch=$E2E_BATCH_SIZE epochs=$E2E_EPOCHS warmup=$E2E_WARMUP_EPOCHS"
    echo "Frozen: batch=$FROZEN_BATCH_SIZE epochs=$FROZEN_EPOCHS warmup=$FROZEN_WARMUP_EPOCHS"
    ;;
  pilot)
    old_seeds=("${SEEDS[@]}"); SEEDS=(42); run_train_set dress; SEEDS=("${old_seeds[@]}")
    ;;
  dress-multiseed)
    run_train_set dress
    ;;
  categories)
    run_train_set "${CATEGORIES[@]}"
    ;;
  modality)
    run_modality
    ;;
  robustness)
    run_robustness
    ;;
  summarize)
    python eval/summarize_supervisor.py --result_root "$RESULT_ROOT" --output_dir "$RESULT_ROOT/tables"
    ;;
  all)
    "$0" preflight
    "$0" categories
    "$0" modality
    "$0" robustness
    "$0" summarize
    ;;
  *)
    echo "Usage: $0 {preflight|pilot|dress-multiseed|categories|modality|robustness|summarize|all}"
    echo "Config: MODELS=$MODELS_CSV SEEDS=$SEEDS_CSV CATEGORIES=$CATEGORIES_CSV"
    echo "Batches: Point=$POINT_BATCH_SIZE E2E/matched=$E2E_BATCH_SIZE Frozen=$FROZEN_BATCH_SIZE; FORCE=$FORCE USE_WANDB=$USE_WANDB"
    exit 2
    ;;
esac
