#!/usr/bin/env bash
# Source from any directory. Never copy a virtualenv between machines.
HUG_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LAVIS_ROOT="${LAVIS_ROOT:-$HUG_PROJECT_ROOT/ref/LAVIS}"
export PYTHONPATH="$HUG_PROJECT_ROOT:$LAVIS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$HUG_PROJECT_ROOT/.migration/cache/huggingface" ]]; then
  export HF_HOME="$HUG_PROJECT_ROOT/.migration/cache/huggingface"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export TRANSFORMERS_CACHE="$HF_HOME/hub"
fi
if [[ -d "$HUG_PROJECT_ROOT/.migration/cache/torch" ]]; then
  export TORCH_HOME="$HUG_PROJECT_ROOT/.migration/cache/torch"
fi
if [[ -f "$HUG_PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$HUG_PROJECT_ROOT/.venv/bin/activate"
elif [[ -f "$LAVIS_ROOT/.venv/bin/activate" ]]; then
  source "$LAVIS_ROOT/.venv/bin/activate"
fi
