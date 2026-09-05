from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any


def import_official_dflash(
    source_dir: str | Path,
    expected_commit: str,
) -> tuple[Any, Any]:
    source = Path(source_dir).resolve()
    if not (source / ".git").is_dir():
        raise RuntimeError(f"DFlash source is not a Git checkout: {source}")
    actual_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"DFlash source commit mismatch: expected {expected_commit}, "
            f"found {actual_commit}"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("dflash.model")
    return module.DFlashDraftModel, module.dflash_generate
