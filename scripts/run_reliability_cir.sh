#!/usr/bin/env bash
set -euo pipefail

# Reliability-aware CIR schema/protocol v2. Point and v1 artifacts are read-only.
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$ROOT_DIR/data/fashion-iq"}
VENV=${VENV:-"$ROOT_DIR/ref/LAVIS/.venv/bin/activate"}
CATEGORY=${CATEGORY:-dress}
SEED=${SEED:-42}
POINT_CHECKPOINT=${POINT_CHECKPOINT:-"$ROOT_DIR/checkpoints/supervisor_protocol_v2/point/$CATEGORY/seed$SEED/checkpoint_best.pth"}
EXPERIMENT_ID=${EXPERIMENT_ID:-reliability_v2}
RESULT_ROOT=${RESULT_ROOT:-"$ROOT_DIR/results/$EXPERIMENT_ID/$CATEGORY/seed$SEED"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$ROOT_DIR/checkpoints/$EXPERIMENT_ID/$CATEGORY/seed$SEED"}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-4}
LAMBDA_FAILURE=${LAMBDA_FAILURE:-1.0}
PATIENCE=${PATIENCE:-3}
FAILURE_K=${FAILURE_K:-10}
MIN_SYNONYM_CHANGED_RATE=${MIN_SYNONYM_CHANGED_RATE:-0.90}
MAX_QUERIES=${MAX_QUERIES:-}

[[ -f "$VENV" ]] && source "$VENV"
cd "$ROOT_DIR"

TRAIN_MANIFEST="$RESULT_ROOT/labels/fiqc_${CATEGORY}_train_seed${SEED}.jsonl"
VAL_MANIFEST="$RESULT_ROOT/labels/fiqc_${CATEGORY}_val_seed${SEED}.jsonl"
BEST_CHECKPOINT="$CHECKPOINT_ROOT/checkpoint_best.pth"
TRAIN_AUDIT="$RESULT_ROOT/labels/fiqc_${CATEGORY}_train_seed${SEED}_corruption_audit.json"
VAL_AUDIT="$RESULT_ROOT/labels/fiqc_${CATEGORY}_val_seed${SEED}_corruption_audit.json"
EVAL_FILE="$RESULT_ROOT/reliability_metrics.json"

max_query_args=()
[[ -n "$MAX_QUERIES" ]] && max_query_args+=(--max_queries "$MAX_QUERIES")

audit_corruptions() {
  local split=$1
  local output="$RESULT_ROOT/labels/fiqc_${CATEGORY}_${split}_seed${SEED}_corruption_audit.json"
  if [[ -f "$output" ]]; then
    echo "SKIP existing corruption audit: $output"
    return
  fi
  mkdir -p "$RESULT_ROOT/labels"
  python -u eval/reliability.py audit-corruptions \
    --data_root "$DATA_ROOT" --split "$split" --category "$CATEGORY" --seed "$SEED" \
    --min_synonym_changed_rate "$MIN_SYNONYM_CHANGED_RATE" --output_file "$output"
}

build_labels() {
  local split=$1
  local manifest="$RESULT_ROOT/labels/fiqc_${CATEGORY}_${split}_seed${SEED}.jsonl"
  local audit="$RESULT_ROOT/labels/fiqc_${CATEGORY}_${split}_seed${SEED}_corruption_audit.json"
  [[ -f "$audit" ]] || { echo "Missing corruption audit; run '$0 audit' first" >&2; exit 1; }
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("schema_version")==2 and d.get("passed") else 1)' "$audit"
  if [[ -f "$manifest" ]]; then
    echo "SKIP existing labels: $manifest"
    return
  fi
  python -u eval/reliability.py build-labels \
    --data_root "$DATA_ROOT" --checkpoint "$POINT_CHECKPOINT" \
    --split "$split" --category "$CATEGORY" --seed "$SEED" --failure_k "$FAILURE_K" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --min_synonym_changed_rate "$MIN_SYNONYM_CHANGED_RATE" \
    --output_dir "$RESULT_ROOT/labels" "${max_query_args[@]}"
}

train_head() {
  [[ -f "$TRAIN_MANIFEST" ]] || { echo "Missing train manifest: $TRAIN_MANIFEST" >&2; exit 1; }
  [[ -f "$VAL_MANIFEST" ]] || { echo "Missing val manifest: $VAL_MANIFEST" >&2; exit 1; }
  if [[ -f "$CHECKPOINT_ROOT/checkpoint_final.pth" ]]; then
    echo "SKIP completed training: $CHECKPOINT_ROOT/checkpoint_final.pth"
    return
  fi
  resume_args=()
  if [[ -f "$CHECKPOINT_ROOT/checkpoint_last.pth" ]]; then
    resume_args+=(--resume_checkpoint "$CHECKPOINT_ROOT/checkpoint_last.pth")
  fi
  mkdir -p "$CHECKPOINT_ROOT" "$RESULT_ROOT/logs"
  python -u train_reliability.py \
    --data_root "$DATA_ROOT" --point_checkpoint "$POINT_CHECKPOINT" \
    --train_manifest "$TRAIN_MANIFEST" --val_manifest "$VAL_MANIFEST" \
    --output_dir "$CHECKPOINT_ROOT" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --num_epochs "$EPOCHS" --lr "$LR" --lambda_failure "$LAMBDA_FAILURE" \
    --patience "$PATIENCE" --seed "$SEED" "${resume_args[@]}" \
    2>&1 | tee -a "$RESULT_ROOT/logs/train_reliability.log"
}

evaluate_head() {
  [[ -f "$BEST_CHECKPOINT" ]] || { echo "Missing best checkpoint: $BEST_CHECKPOINT" >&2; exit 1; }
  if [[ -f "$EVAL_FILE" ]]; then
    echo "SKIP existing evaluation: $EVAL_FILE"
    return
  fi
  python -u eval/reliability.py evaluate \
    --data_root "$DATA_ROOT" --manifest "$VAL_MANIFEST" --train_manifest "$TRAIN_MANIFEST" \
    --checkpoint "$BEST_CHECKPOINT" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --min_synonym_changed_rate "$MIN_SYNONYM_CHANGED_RATE" --output_file "$EVAL_FILE"
}

check_gate() {
  [[ -f "$EVAL_FILE" ]] || { echo "Missing evaluation: $EVAL_FILE" >&2; exit 1; }
  python -c 'import json,sys; gate=json.load(open(sys.argv[1]))["adaptive_fusion_gate"]; print(json.dumps(gate, indent=2)); raise SystemExit(0 if gate.get("passed") else 1)' "$EVAL_FILE"
}

case "${1:-help}" in
  preflight)
    python -m py_compile modules/fiqc.py modules/reliability.py models/reliability_model.py \
      eval/reliability.py train_reliability.py
    bash -n scripts/run_reliability_cir.sh
    PYTHONPATH=. pytest -q tests/test_reliability.py
    [[ -f "$POINT_CHECKPOINT" ]] || { echo "Missing Point checkpoint: $POINT_CHECKPOINT" >&2; exit 1; }
    for split in train val; do
      [[ -f "$DATA_ROOT/captions/cap.$CATEGORY.$split.json" ]] || {
        echo "Missing Fashion-IQ captions for $CATEGORY/$split" >&2; exit 1;
      }
    done
    echo "Preflight passed: category=$CATEGORY seed=$SEED failure_k=$FAILURE_K"
    ;;
  audit) audit_corruptions train; audit_corruptions val ;;
  gate) check_gate ;;
  build-train) build_labels train ;;
  build-val) build_labels val ;;
  train) train_head ;;
  evaluate) evaluate_head ;;
  pilot)
    "$0" preflight
    "$0" audit
    "$0" build-train
    "$0" build-val
    "$0" train
    "$0" evaluate
    "$0" gate
    ;;
  *)
    echo "Usage: $0 {preflight|audit|build-train|build-val|train|evaluate|gate|pilot}"
    echo "No phase overwrites an existing manifest, evaluation, or completed checkpoint."
    exit 2
    ;;
esac
