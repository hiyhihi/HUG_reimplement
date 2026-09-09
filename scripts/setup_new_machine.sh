#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
[[ -f .migration/environment.txt && -d ref/LAVIS/lavis ]] || {
  echo 'Restore the migration bundle first (see README.md).'; exit 2;
}
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 2; }
if [[ ! -e .venv ]]; then
  uv venv --python 3.8 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,8), "Use Python 3.8 for this locked experiment"'
# Snapshot includes transitive dependencies. --no-deps preserves the working environment,
# even where the old LAVIS package metadata has incompatible dependency declarations.
uv pip install --python .venv/bin/python --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r .migration/environment.txt
source scripts/project_env.sh
python -c 'import torch, transformers, lavis, data; print("torch", torch.__version__, "CUDA", torch.cuda.is_available()); print("transformers", transformers.__version__)'
PYTHONPATH=. python -m pytest -q tests/test_migration.py tests/test_reliability.py
bash scripts/run_reliability_cir.sh preflight
echo 'Setup/CPU preflight complete. Before long training: check nvidia-smi and the README GPU smoke test.'
