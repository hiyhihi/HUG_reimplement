"""Relocate trusted artifact paths without rewriting checkpoints or manifests."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_artifact_path(value, root=None):
    root = Path(root or PROJECT_ROOT).resolve()
    source = os.environ.get("HUG_SOURCE_ROOT")
    metadata = root / ".migration" / "manifest.json"
    if not source and metadata.is_file():
        source = json.loads(metadata.read_text())["source_root"]
    path = Path(value)
    if path.is_absolute() and source:
        try:
            path = root / path.relative_to(Path(source))
        except ValueError:
            pass  # No basename guessing: a different backbone must stay different.
    elif not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing artifact: {path}. Restore the migration bundle, or set "
            "HUG_SOURCE_ROOT to its original project root. Do not substitute another checkpoint."
        )
    return path
