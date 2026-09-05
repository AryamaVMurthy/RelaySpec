from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from relayspec.submission_plan import PlanError, validate_plan


@pytest.fixture
def plan(tmp_path):
    (tmp_path / "evidence.txt").write_text("fixture evidence")
    result = {
        "schema_version": 1,
        "execution_plan": "evidence.txt",
        "framing_plan": "evidence.txt",
        "review_study": "evidence.txt",
        "audit": "evidence.txt",
        "issue_ids": [f"R{i}" for i in range(1, 19)],
        "tasks": [
            {
                "id": "A",
                "status": "planned",
                "required": True,
                "depends_on": [],
                "issues": [f"R{i}" for i in range(1, 19)],
                "existing_files": ["evidence.txt"],
                "completion_record": "A.json",
            },
            {
                "id": "B",
                "status": "planned",
                "required": False,
                "depends_on": ["A"],
                "issues": [],
                "completion_record": "B.json",
            },
        ],
    }
    return result


def complete_a(plan, root):
    plan["tasks"][0]["status"] = "complete"
    digest = hashlib.sha256((root / "evidence.txt").read_bytes()).hexdigest()
    (root / "A.json").write_text(
        json.dumps(
            {
                "task_id": "A",
                "summary": "Reviewed fixture result",
                "evidence": [{"path": "evidence.txt", "sha256": digest}],
            }
        )
    )


def test_next_work_and_readiness_require_actual_completion(plan, tmp_path):
    assert validate_plan(plan, tmp_path)["next_tasks"] == ["A"]
    assert validate_plan(plan, tmp_path)["remaining_required"] == ["A"]
    complete_a(plan, tmp_path)
    status = validate_plan(plan, tmp_path)
    assert status["next_tasks"] == ["B"]
    assert status["remaining_required"] == []


def test_stale_evidence_invalidates_completion(plan, tmp_path):
    complete_a(plan, tmp_path)
    (tmp_path / "evidence.txt").write_text("changed after review")
    with pytest.raises(PlanError, match="stale evidence"):
        validate_plan(plan, tmp_path)


def test_complete_without_record_fails(plan, tmp_path):
    plan["tasks"][0]["status"] = "complete"
    with pytest.raises(PlanError, match="missing"):
        validate_plan(plan, tmp_path)


def test_complete_before_dependencies_fails(plan, tmp_path):
    plan["tasks"][1]["status"] = "complete"
    with pytest.raises(PlanError, match="unfinished dependencies"):
        validate_plan(plan, tmp_path)


@pytest.mark.parametrize("defect", ["cycle", "unknown", "coverage", "defer"])
def test_invalid_graph_or_issue_coverage_fails(plan, tmp_path, defect):
    broken = copy.deepcopy(plan)
    if defect == "cycle":
        broken["tasks"][0]["depends_on"] = ["B"]
    elif defect == "unknown":
        broken["tasks"][1]["depends_on"] = ["MISSING"]
    elif defect == "coverage":
        broken["tasks"][0]["issues"].remove("R18")
    else:
        broken["tasks"][0]["status"] = "deferred"
    with pytest.raises(PlanError):
        validate_plan(broken, tmp_path)


def test_actual_repository_plan_is_consistent():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/submission/plan.json").read_text())
    validate_plan(plan, root)
