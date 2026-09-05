from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.manuscript

ROOT = Path(__file__).resolve().parents[1]


def _load_auditor():
    path = ROOT / "scripts" / "audit_iclr_manuscript.py"
    spec = importlib.util.spec_from_file_location("audit_iclr_manuscript", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_audit_passes() -> None:
    auditor = _load_auditor()
    result = auditor.audit_manuscript(ROOT, write_report=False)
    failures = [check for check in result["checks"] if not check["passed"]]
    assert failures == []
    assert result["main_text_end_page"] <= 9
    assert result["citation_count"] >= 15


def test_paper_is_anonymous_and_uses_generated_results() -> None:
    auditor = _load_auditor()
    result = auditor.audit_manuscript(ROOT, write_report=False)
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["official ICLR 2027 style"]["passed"]
    assert checks["anonymous review source"]["passed"]
    assert checks["generated result assets"]["passed"]
    assert checks["prompt separation audits"]["passed"]
    assert checks["embedded fonts and PDF parser"]["passed"]
    assert checks["recorded visual review"]["passed"]
    assert checks["language constraints"]["passed"]
    assert checks["clean anonymous status header"]["passed"]
    assert checks["reader-facing scientific detail"]["passed"]


def test_all_citations_resolve() -> None:
    auditor = _load_auditor()
    result = auditor.audit_manuscript(ROOT, write_report=False)
    assert result["missing_citations"] == []
