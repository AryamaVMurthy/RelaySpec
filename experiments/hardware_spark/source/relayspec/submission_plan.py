"""Check research-plan consistency; never run experiments or judge their results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class PlanError(ValueError):
    """A plan or its claimed completion evidence is inconsistent."""


def _local_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PlanError(f"expected a repository-relative file: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise PlanError(f"missing or outside-repository file: {relative}")
    return path


def validate_plan(plan: dict[str, Any], root: Path) -> dict[str, list[str]]:
    """Validate files, dependency graph, issue coverage and completion hashes."""
    if plan.get("schema_version") != 1:
        raise PlanError("unsupported schema_version")
    for key in ("execution_plan", "framing_plan", "review_study", "audit"):
        _local_file(root, plan.get(key))
    expected_issues = {f"R{i}" for i in range(1, 19)}
    if set(plan.get("issue_ids", [])) != expected_issues:
        raise PlanError("issue_ids must cover audit issues R1 through R18")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise PlanError("tasks must be a nonempty list")
    indexed = {}
    for task in tasks:
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id or task_id in indexed:
            raise PlanError(f"missing or duplicate task id: {task_id!r}")
        indexed[task_id] = task
    covered = set()
    for task_id, task in indexed.items():
        if task.get("status") not in {"planned", "in_progress", "complete", "deferred"}:
            raise PlanError(f"{task_id}: invalid status")
        if not isinstance(task.get("required"), bool):
            raise PlanError(f"{task_id}: required must be boolean")
        if task["required"] and task["status"] == "deferred":
            raise PlanError(f"{task_id}: required tasks cannot be silently deferred")
        dependencies = task.get("depends_on")
        if not isinstance(dependencies, list) or any(
            dep not in indexed for dep in dependencies
        ):
            raise PlanError(f"{task_id}: unknown or invalid dependencies")
        issues = task.get("issues")
        if not isinstance(issues, list) or not set(issues) <= expected_issues:
            raise PlanError(f"{task_id}: invalid issue references")
        covered.update(issues)
        for relative in task.get("existing_files", []):
            _local_file(root, relative)
    if covered != expected_issues:
        raise PlanError(f"unassigned audit issues: {sorted(expected_issues - covered)}")

    visiting, visited = set(), set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise PlanError(f"dependency cycle at {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in indexed[task_id]["depends_on"]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in indexed:
        visit(task_id)

    for task_id, task in indexed.items():
        if task["status"] != "complete":
            continue
        if any(indexed[d]["status"] != "complete" for d in task["depends_on"]):
            raise PlanError(f"{task_id}: completed task has unfinished dependencies")
        record_path = _local_file(root, task.get("completion_record"))
        record = json.loads(record_path.read_text())
        if record.get("task_id") != task_id or not record.get("summary"):
            raise PlanError(f"{task_id}: completion record needs task_id and summary")
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise PlanError(f"{task_id}: completion requires evidence files")
        for item in evidence:
            path = _local_file(root, item.get("path"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item.get("sha256"):
                raise PlanError(f"{task_id}: stale evidence hash: {item.get('path')}")

    return {
        "remaining_required": [
            t["id"] for t in tasks if t["required"] and t["status"] != "complete"
        ],
        "next_tasks": [
            t["id"]
            for t in tasks
            if t["status"] in {"planned", "in_progress"}
            and all(indexed[d]["status"] == "complete" for d in t["depends_on"])
        ],
        "deferred": [t["id"] for t in tasks if t["status"] == "deferred"],
    }
