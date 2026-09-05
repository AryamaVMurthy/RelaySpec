from __future__ import annotations

import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PYTHON_FENCE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str:
    match = _PYTHON_FENCE.search(completion) or _ANY_FENCE.search(completion)
    return (match.group(1) if match else completion).strip()


def _set_capped_limit(kind: int, requested: int) -> None:
    """Lower a resource limit without attempting to raise a cluster cap."""
    _, hard = resource.getrlimit(kind)
    cap = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    resource.setrlimit(kind, (cap, cap))


def _limit_child(timeout_seconds: int) -> None:
    _set_capped_limit(resource.RLIMIT_CORE, 0)
    _set_capped_limit(resource.RLIMIT_CPU, timeout_seconds + 1)
    _set_capped_limit(resource.RLIMIT_AS, 1_073_741_824)
    _set_capped_limit(resource.RLIMIT_FSIZE, 1_048_576)
    _set_capped_limit(resource.RLIMIT_NOFILE, 64)


def run_python_tests(
    code: str,
    tests: list[str],
    *,
    timeout_seconds: int = 5,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("code-test timeout must be positive")
    program = code.rstrip() + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory(prefix="relayspec-code-eval-") as directory:
        candidate = Path(directory) / "candidate.py"
        candidate.write_text(program, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-I", str(candidate)],
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            # The official paper result uses EvalPlus. This fallback additionally
            # installs POSIX limits in each isolated scoring child.
            preexec_fn=lambda: _limit_child(timeout_seconds),  # noqa: PLW1509
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds + 2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return {
                "passed": False,
                "status": "timeout",
                "returncode": process.returncode,
                "stdout": stdout[-2000:],
                "stderr": stderr[-2000:],
            }
    if process.returncode == 0:
        status = "pass"
    elif process.returncode is not None and process.returncode < 0:
        status = "signal"
    else:
        status = "fail"
    return {
        "passed": process.returncode == 0,
        "status": status,
        "returncode": process.returncode,
        "stdout": stdout[-2000:],
        "stderr": stderr[-2000:],
    }
