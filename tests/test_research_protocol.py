from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from relayspec.research_protocol import (
    ProtocolError,
    audit_config,
    load_protocol,
    validate_protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def test_repository_protocol_is_complete_and_valid() -> None:
    protocol = load_protocol(ROOT / "configs/relayspec_protocol.yaml")

    validate_protocol(protocol, repository_root=ROOT)


def test_frozen_decision_requires_primary_locator_or_derivation() -> None:
    protocol = {
        "protocol_version": 1,
        "decisions": [
            {
                "id": "unsupported",
                "value": 0.1,
                "evidence_class": "P",
                "status": "frozen",
                "scope": "loss",
            }
        ],
        "controlled_config_paths": [],
    }

    with pytest.raises(ProtocolError, match="source_locator or derivation"):
        validate_protocol(protocol, repository_root=ROOT)


def test_active_config_rejects_unregistered_scientific_constant(tmp_path: Path) -> None:
    config_path = tmp_path / "active.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "protocol_version": 1,
                "generation": {"max_new_tokens": 2048},
                "relay_training": {"mystery_weight": 0.37},
            }
        )
    )
    protocol = {
        "protocol_version": 1,
        "decisions": [
            {
                "id": "generation.max_new_tokens",
                "value": 2048,
                "evidence_class": "P",
                "status": "frozen",
                "scope": "generation",
                "source_locator": "chen2026dflash:Table 1",
            }
        ],
        "controlled_config_paths": [
            {
                "pattern": "generation.max_new_tokens",
                "decision_id": "generation.max_new_tokens",
            }
        ],
        "administrative_config_paths": ["protocol_version"],
        "administrative_path_rationales": {"protocol_version": "schema selector only"},
    }

    errors = audit_config(config_path, protocol)

    assert errors == [
        "unregistered numeric constant: relay_training.mystery_weight=0.37"
    ]


def test_active_config_checks_frozen_value(tmp_path: Path) -> None:
    config_path = tmp_path / "active.yaml"
    config_path.write_text(
        yaml.safe_dump({"protocol_version": 1, "generation": {"max_new_tokens": 96}})
    )
    protocol = {
        "protocol_version": 1,
        "decisions": [
            {
                "id": "generation.max_new_tokens",
                "value": 2048,
                "evidence_class": "P",
                "status": "frozen",
                "scope": "generation",
                "source_locator": "chen2026dflash:Table 1",
            }
        ],
        "controlled_config_paths": [
            {
                "pattern": "generation.max_new_tokens",
                "decision_id": "generation.max_new_tokens",
            }
        ],
        "administrative_config_paths": ["protocol_version"],
        "administrative_path_rationales": {"protocol_version": "schema selector only"},
    }

    errors = audit_config(config_path, protocol)

    assert errors == [
        "frozen decision generation.max_new_tokens requires 2048, found 96"
    ]


def test_validation_candidate_allows_declared_values(tmp_path: Path) -> None:
    config_path = tmp_path / "active.yaml"
    config_path.write_text(
        yaml.safe_dump({"protocol_version": 1, "relay_training": {"rank": 512}})
    )
    protocol = {
        "protocol_version": 1,
        "decisions": [
            {
                "id": "relay.rank",
                "value": [256, 512, 1024, "full"],
                "evidence_class": "V",
                "status": "pending_validation",
                "scope": "relay architecture",
                "source_locator": "reports/design-selection/rank.json",
            }
        ],
        "controlled_config_paths": [
            {"pattern": "relay_training.rank", "decision_id": "relay.rank"}
        ],
        "administrative_config_paths": ["protocol_version"],
        "administrative_path_rationales": {"protocol_version": "schema selector only"},
    }

    assert audit_config(config_path, protocol) == []


def test_administrative_exemptions_require_machine_checked_rationales() -> None:
    protocol = {
        "protocol_version": 1,
        "decisions": [
            {
                "id": "generation.max_new_tokens",
                "value": 2048,
                "evidence_class": "P",
                "status": "frozen",
                "scope": "generation",
                "source_locator": "chen2026dflash:Table 1",
            }
        ],
        "controlled_config_paths": [],
        "administrative_config_paths": ["protocol_version", "logging.*"],
        "administrative_path_rationales": {
            "protocol_version": "schema selector only",
        },
    }

    with pytest.raises(ProtocolError, match=r"logging\.\*.*rationale"):
        validate_protocol(protocol, repository_root=ROOT)


def test_repository_protocol_backs_every_administrative_exemption() -> None:
    protocol = load_protocol(ROOT / "configs/relayspec_protocol.yaml")

    exemptions = protocol["administrative_config_paths"]
    rationales = protocol["administrative_path_rationales"]

    assert set(exemptions) == set(rationales)
    assert all(
        isinstance(rationales[path], str) and rationales[path] for path in exemptions
    )
