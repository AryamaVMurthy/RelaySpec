from __future__ import annotations

from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

EVIDENCE_CLASSES = {"F", "P", "O", "V", "T", "A"}
DECISION_STATUSES = {"frozen", "pending_validation", "historical"}


class ProtocolError(ValueError):
    """Raised when the research protocol is incomplete or inconsistent."""


def load_protocol(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ProtocolError("protocol root must be a mapping")
    return payload


def _decision_index(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    decisions = protocol.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ProtocolError("protocol decisions must be a non-empty list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ProtocolError("each decision must be a mapping")
        decision_id = decision.get("id")
        if not isinstance(decision_id, str) or not decision_id:
            raise ProtocolError("each decision requires a non-empty id")
        if decision_id in indexed:
            raise ProtocolError(f"duplicate decision id: {decision_id}")
        indexed[decision_id] = decision
    return indexed


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> None:
    if not isinstance(protocol.get("protocol_version"), int):
        raise ProtocolError("protocol_version must be an integer")
    decisions = _decision_index(protocol)
    for decision_id, decision in decisions.items():
        evidence_class = decision.get("evidence_class")
        if evidence_class not in EVIDENCE_CLASSES:
            raise ProtocolError(
                f"decision {decision_id} has invalid evidence_class {evidence_class!r}"
            )
        status = decision.get("status")
        if status not in DECISION_STATUSES:
            raise ProtocolError(f"decision {decision_id} has invalid status {status!r}")
        if not isinstance(decision.get("scope"), str) or not decision["scope"]:
            raise ProtocolError(f"decision {decision_id} requires a scope")
        if "value" not in decision:
            raise ProtocolError(f"decision {decision_id} requires a value")
        if evidence_class != "A" and not (
            decision.get("source_locator") or decision.get("derivation")
        ):
            raise ProtocolError(
                f"decision {decision_id} requires source_locator or derivation"
            )
        if evidence_class == "A" and not decision.get("rationale"):
            raise ProtocolError(
                f"administrative decision {decision_id} requires a rationale"
            )

    rules = protocol.get("controlled_config_paths")
    if not isinstance(rules, list):
        raise ProtocolError("controlled_config_paths must be a list")
    for rule in rules:
        if not isinstance(rule, Mapping) or not isinstance(rule.get("pattern"), str):
            raise ProtocolError("each controlled config path requires a pattern")
        if rule.get("decision_id") not in decisions:
            raise ProtocolError(
                f"unknown decision_id in controlled path: {rule.get('decision_id')}"
            )

    root = Path(repository_root)
    for relative in protocol.get("required_artifacts", []):
        if not (root / relative).exists():
            raise ProtocolError(f"required protocol artifact is missing: {relative}")

    active_globs = protocol.get("active_config_globs", [])
    if not isinstance(active_globs, list):
        raise ProtocolError("active_config_globs must be a list")

    administrative = protocol.get("administrative_config_paths", [])
    rationales = protocol.get("administrative_path_rationales", {})
    if not isinstance(administrative, list):
        raise ProtocolError("administrative_config_paths must be a list")
    if not isinstance(rationales, Mapping):
        raise ProtocolError("administrative_path_rationales must be a mapping")
    for pattern in administrative:
        rationale = rationales.get(pattern)
        if not isinstance(rationale, str) or not rationale.strip():
            raise ProtocolError(
                f"administrative path {pattern} requires a non-empty rationale"
            )
    undeclared_rationales = set(rationales) - set(administrative)
    if undeclared_rationales:
        raise ProtocolError(
            "administrative rationales without matching exemptions: "
            + ", ".join(sorted(undeclared_rationales))
        )

    for pattern in active_globs:
        for config_path in root.glob(pattern):
            errors = audit_config(config_path, protocol)
            if errors:
                joined = "; ".join(errors)
                raise ProtocolError(f"{config_path.relative_to(root)}: {joined}")


def _numeric_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, int | float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(child, child_prefix)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from _numeric_leaves(child, child_prefix)


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(path, pattern) for pattern in patterns)


def _value_matches(observed: float, expected: Any) -> bool:
    if isinstance(expected, list):
        return observed in expected
    return observed == expected


def audit_config(
    config_path: str | Path,
    protocol: Mapping[str, Any],
) -> list[str]:
    payload = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(payload, Mapping):
        return ["config root must be a mapping"]
    expected_version = protocol.get("protocol_version")
    if payload.get("protocol_version") != expected_version:
        return [
            (
                "protocol_version mismatch: "
                f"expected {expected_version}, found {payload.get('protocol_version')!r}"
            )
        ]

    decisions = _decision_index(protocol)
    rules = protocol.get("controlled_config_paths", [])
    administrative = protocol.get("administrative_config_paths", [])
    errors: list[str] = []
    for path, observed in _numeric_leaves(payload):
        if _matches(path, administrative):
            continue
        matching_rules = [rule for rule in rules if fnmatchcase(path, rule["pattern"])]
        if not matching_rules:
            errors.append(f"unregistered numeric constant: {path}={observed}")
            continue
        if len(matching_rules) > 1:
            errors.append(f"ambiguous protocol rules for {path}")
            continue
        decision = decisions[matching_rules[0]["decision_id"]]
        if decision["status"] == "frozen" and not _value_matches(
            observed, decision["value"]
        ):
            errors.append(
                f"frozen decision {decision['id']} requires {decision['value']}, "
                f"found {observed}"
            )
        if decision["status"] == "pending_validation" and not _value_matches(
            observed, decision["value"]
        ):
            errors.append(
                f"validation decision {decision['id']} permits {decision['value']}, "
                f"found {observed}"
            )
    return errors
