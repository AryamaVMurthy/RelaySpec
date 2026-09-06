"""Reproduce saved quality scoring without modifying original experiment artifacts."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def verify_saved_scores(run, scoring_repo):
    run, scoring_repo = Path(run), Path(scoring_repo)
    if not (scoring_repo / "scripts/analyze_controlled_run.py").is_file():
        raise ValueError("pinned scoring repository is unavailable")
    with tempfile.TemporaryDirectory(prefix="relayspec-quality-replay-") as temporary:
        fresh = Path(temporary)
        for path in [
            run / "completion-gate.json",
            *sorted(run.glob("benchmark-rank*.jsonl")),
        ]:
            shutil.copy2(path, fresh / path.name)
        subprocess.run(
            [sys.executable, "scripts/analyze_controlled_run.py", str(fresh)],
            cwd=scoring_repo,
            env={**os.environ, "PYTHONPATH": "src:vendor/qwen-score-deps"},
            check=True,
            capture_output=True,
        )
        if json.loads((fresh / "analysis.json").read_text()) != json.loads(
            (run / "analysis.json").read_text()
        ):
            raise ValueError(
                "quality analysis does not reproduce from pinned raw scoring"
            )
        if (fresh / "math-scored.jsonl").read_bytes() != (
            run / "math-scored.jsonl"
        ).read_bytes():
            raise ValueError("quality scores do not reproduce from raw text")
