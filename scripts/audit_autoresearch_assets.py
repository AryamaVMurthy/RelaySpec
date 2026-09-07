"""Rebuild autoresearch tables and raster plots in an isolated temporary copy."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BUILDERS = [
    "capture_memory", "compact_composition", "compression_objective",
    "crossdomain_depth", "downscale", "downscale_depth", "downscale_nonlinearity",
    "early_nonlinearity", "head_extension", "head_precision", "interface_confirmation",
    "interface_profile", "layer_depth", "native_student", "precision_breadth",
    "research_screen", "small_record_paper", "rate_check", "native_code", "compact_interface_data", "native_initialization", "native_correction", "native_eagle", "sampling_breadth", "native_eagle_code", "capacity14_full_answer", "native_activation",
]


def audit(root):
    root = Path(root).resolve()
    with tempfile.TemporaryDirectory(prefix="relayspec-autoresearch-audit-") as directory:
        sandbox = Path(directory)
        for name in ("reports", "configs", "scripts", "src"):
            shutil.copytree(root / name, sandbox / name)
        paper = sandbox / "paper/iclr2027"
        for name in ("generated", "figures"):
            (paper / name).mkdir(parents=True)
        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        for name in BUILDERS:
            result = subprocess.run(
                [sys.executable, str(sandbox / f"scripts/build_{name}_assets.py")],
                cwd=sandbox, env=env, capture_output=True, text=True, timeout=180,
            )
            if result.returncode:
                return False, f"{name} regeneration failed: {result.stderr[-1200:]}"
        checked = []
        for path in sorted(paper.rglob("*")):
            # PDF creation timestamps and provenance absolute paths are not
            # byte-stable. Compare actual table text and raster plot pixels.
            if path.suffix not in {".tex", ".png"}:
                continue
            relative = path.relative_to(paper)
            actual = root / "paper/iclr2027" / relative
            if not actual.exists() or actual.read_bytes() != path.read_bytes():
                return False, f"stale or missing autoresearch asset: {relative}"
            checked.append(str(relative))
        if not checked:
            return False, "No autoresearch assets checked"
        return True, f"{len(checked)} autoresearch tables/raster plots reproduce from {len(BUILDERS)} isolated builders"


if __name__ == "__main__":
    passed, evidence = audit(Path(__file__).resolve().parents[1])
    print(json.dumps({"passed": passed, "evidence": evidence}))
    raise SystemExit(0 if passed else 1)
