#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MODELS=point,hug_frozen_point
export RESULT_ROOT=${RESULT_ROOT:-"$ROOT_DIR/results/supervisor_protocol_v2"}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$ROOT_DIR/checkpoints/supervisor_protocol_v2"}

case "${1:-pilot}" in
  pilot)
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" preflight
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" pilot
    ;;
  clean)
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" preflight
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" categories
    ;;
  modality)
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" modality
    ;;
  robustness)
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" robustness
    ;;
  evaluate)
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" modality
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" robustness
    ;;
  all)
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" preflight
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" categories
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" modality
    "$ROOT_DIR/scripts/run_supervisor_experiments.sh" robustness
    ;;
  *)
    echo "Usage: $0 {pilot|clean|modality|robustness|evaluate|all}" >&2
    exit 2
    ;;
esac
