from __future__ import annotations

from relayspec.benchmarking import resolve_revision


def test_resolve_revision_keeps_revision_for_a_hub_id() -> None:
    assert resolve_revision("Qwen/Qwen3-8B", "abc123") == "abc123"


def test_resolve_revision_drops_revision_for_an_existing_local_directory(
    tmp_path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint-step-50"
    checkpoint_dir.mkdir()
    assert resolve_revision(str(checkpoint_dir), "main") is None


def test_resolve_revision_keeps_revision_when_none_given() -> None:
    assert resolve_revision("Qwen/Qwen3-8B", None) is None
