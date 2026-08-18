#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET=${DATASET:-fashion-iq}
DATA_ROOT=${DATA_ROOT:-$ROOT_DIR/data/fashion-iq}
CATEGORY=${CATEGORY:-dress}
RESULT_ROOT=${RESULT_ROOT:-$ROOT_DIR/results/person2}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
VENV=${VENV:-$ROOT_DIR/ref/LAVIS/.venv/bin/activate}
[[ -f "$VENV" ]] && source "$VENV"
cd "$ROOT_DIR"

run_sweep() {
  local model_id=$1 checkpoint=$2 skip=${3:-false} phase=${4:-week3}
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; return 1; }
  local extra=() modes=(mean)
  [[ "$skip" == true ]] && extra+=(--skip_uncertainty)
  if [[ "$skip" == false ]]; then extra+=(--include_mismatch); modes+=(probabilistic); fi
  python -u eval/robustness.py sweep --dataset "$DATASET" --data_root "$DATA_ROOT" --split val --category "$CATEGORY" \
    --checkpoint "$checkpoint" --model_id "$model_id" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --seed "$SEED" --distance_modes "${modes[@]}" \
    --output_dir "$RESULT_ROOT/$phase/$DATASET/$CATEGORY/$model_id" "${extra[@]}"
}

case "${1:-help}" in
  week1)
    python -m py_compile eval/robustness.py
    for severity in 0 1 2 3 4; do
      python eval/robustness.py generate --data_root "$DATA_ROOT" --group A --modality image --corruption gaussian_blur --severity "$severity" --seed "$SEED" --output_file "$RESULT_ROOT/week1/A_image_blur_s${severity}.json"
      python eval/robustness.py generate --data_root "$DATA_ROOT" --group A --modality text --corruption typo --severity "$severity" --seed "$SEED" --output_file "$RESULT_ROOT/week1/A_text_typo_s${severity}.json"
      python eval/robustness.py generate --data_root "$DATA_ROOT" --group B --modality image --corruption occlusion --severity "$severity" --seed "$SEED" --output_file "$RESULT_ROOT/week1/B_image_occlusion_s${severity}.json"
      python eval/robustness.py generate --data_root "$DATA_ROOT" --group B --modality text --corruption token_dropout --severity "$severity" --seed "$SEED" --output_file "$RESULT_ROOT/week1/B_text_dropout_s${severity}.json"
    done
    shopt -s nullglob
    clean_manifests=("$RESULT_ROOT"/week1/*_s0.json)
    [[ ${#clean_manifests[@]} -eq 4 ]] || {
      echo "Expected 4 level-0 manifests, found ${#clean_manifests[@]}" >&2; exit 1;
    }
    for clean_manifest in "${clean_manifests[@]}"; do
      if grep -Fq '"changed": true' "$clean_manifest"; then
        echo "Level-0 identity test failed: $clean_manifest" >&2; exit 1
      else
        grep_status=$?
        [[ $grep_status -eq 1 ]] || {
          echo "Could not validate level-0 manifest: $clean_manifest" >&2; exit 1;
        }
      fi
    done
    python eval/robustness.py generate --data_root "$DATA_ROOT" --group A --modality text --corruption typo --severity 2 --seed "$SEED" --output_file /tmp/person2_determinism.json
    cmp "$RESULT_ROOT/week1/A_text_typo_s2.json" /tmp/person2_determinism.json
    echo "Week-1 tests passed: level-0 identity and deterministic seed."
    ;;
  week2)
    run_sweep paper_dress "${PAPER_CHECKPOINT:-$ROOT_DIR/checkpoints/paper_dress_seed42/checkpoint_best.pth}" false week2
    python eval/robustness.py aggregate --input_glob "$RESULT_ROOT/week2/**/*.json" --output_file "$RESULT_ROOT/week2_summary.json"
    ;;
  week3)
    run_sweep point_dress "${POINT_CHECKPOINT:-$ROOT_DIR/checkpoints/point_dress_seed42/checkpoint_best.pth}" true week3
    run_sweep legacy_dress "${LEGACY_CHECKPOINT:-$ROOT_DIR/checkpoints/legacy_dress_seed42/checkpoint_final.pth}" false week3
    run_sweep paper_dress "${PAPER_CHECKPOINT:-$ROOT_DIR/checkpoints/paper_dress_seed42/checkpoint_best.pth}" false week3
    python eval/robustness.py aggregate --input_glob "$RESULT_ROOT/week3/**/*.json" --output_file "$RESULT_ROOT/week3_summary.json"
    ;;
  week4-5)
    stage=${STAGE:-u1}
    [[ "$stage" == u1 || "$stage" == u2 || "$stage" == u2_heads ]] || { echo "For week4-5, set STAGE=u1, STAGE=u2, or STAGE=u2_heads" >&2; exit 2; }
    "$ROOT_DIR/scripts/run_calibrated_hug.sh" "$stage" "$CATEGORY"
    checkpoint="$ROOT_DIR/checkpoints/${stage}_${CATEGORY}_seed${SEED}/checkpoint_best.pth"
    [[ "$stage" == u1 ]] && checkpoint="$ROOT_DIR/checkpoints/${stage}_${CATEGORY}_seed${SEED}/checkpoint_final.pth"
    run_sweep "${stage}_${CATEGORY}" "$checkpoint" false "$stage"
    python eval/robustness.py aggregate --input_glob "$RESULT_ROOT/$stage/**/*.json" --output_file "$RESULT_ROOT/${stage}_summary.json"
    echo "${stage} training and robustness evaluation completed: $RESULT_ROOT/${stage}_summary.json"
    ;;
  week6)
    "$ROOT_DIR/scripts/run_calibrated_hug.sh" u3 "$CATEGORY"
    checkpoint="$ROOT_DIR/checkpoints/u3_${CATEGORY}_seed${SEED}/checkpoint_best.pth"
    run_sweep "u3_${CATEGORY}" "$checkpoint" false u3
    python eval/robustness.py aggregate --input_glob "$RESULT_ROOT/u3/**/*.json" --output_file "$RESULT_ROOT/u3_summary.json"
    echo "u3 training and robustness evaluation completed: $RESULT_ROOT/u3_summary.json"
    ;;
  week7-8)
    echo "Dùng: $0 custom MODEL_ID CHECKPOINT DATASET CATEGORY DATA_ROOT [point]"; exit 2
    ;;
  week9-10)
    python eval/robustness.py aggregate --input_glob "$RESULT_ROOT/**/*.json" --output_file "$RESULT_ROOT/final_summary.json"
    ;;
  custom)
    model_id=${2:?MODEL_ID}; checkpoint=${3:?CHECKPOINT}; DATASET=${4:?DATASET}; CATEGORY=${5:?CATEGORY}; DATA_ROOT=${6:?DATA_ROOT}; point=${7:-hug}
    skip=false; [[ "$point" == point ]] && skip=true
    run_sweep "$model_id" "$checkpoint" "$skip" week7-8
    ;;
  *)
    echo "Usage: $0 {week1|week2|week3|week4-5|week6|week7-8|week9-10|custom}"; exit 2
    ;;
esac
