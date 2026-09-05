"""The breadth paired-AR builder must compute the same statistic as the rest
of the pipeline.

These tests run against the archived ``dflash-8b-breadth`` run, whose
relay-over-source-reuse ratios are already published in the manuscript's full
workload table. If this builder reproduces those numbers from the raw per-rank
rows, then the plain-AR ratios it produces from the reruns are computed the
same way, and the breadth figure cannot silently drift from the tables.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports" / "final" / "dflash-8b-breadth"


def _module():
    spec = importlib.util.spec_from_file_location(
        "build_breadth_paired_ar", ROOT / "scripts" / "build_breadth_paired_ar.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not RUN.exists(), reason="archived breadth run is not present"
)


def test_mtbench_clusters_by_conversation_not_by_turn():
    """MT-Bench turns share a prefix, so the bootstrap unit is the conversation.

    Resampling the 160 turns independently would understate the interval.
    The rest of the paper bootstraps 80 conversations, and so must this.
    """
    module = _module()
    rows = module._load_rows(RUN)
    mtbench = [row for row in rows if row["benchmark"] == "mtbench"]
    assert len({row["problem_id"] for row in mtbench}) == 160
    assert len({module._cluster_key(row) for row in mtbench}) == 80


@pytest.mark.parametrize(
    ("benchmark", "clusters"),
    [("gsm8k", 128), ("humaneval", 164), ("mbpp", 378), ("mtbench", 80)],
)
def test_cluster_counts_match_the_published_protocol(benchmark, clusters):
    module = _module()
    rows = [row for row in module._load_rows(RUN) if row["benchmark"] == benchmark]
    assert len({module._cluster_key(row) for row in rows}) == clusters


@pytest.mark.parametrize(
    ("benchmark", "published"),
    [
        ("gsm8k", 1.424),
        ("humaneval", 1.220),
        ("mbpp", 1.287),
        ("mtbench", 1.341),
    ],
)
def test_reproduces_the_published_relay_over_source_reuse_ratios(benchmark, published):
    module = _module()
    rows = [row for row in module._load_rows(RUN) if row["benchmark"] == benchmark]
    result = module._speedup(rows, "optimized_source_reuse", "relay_p")
    assert result["estimate"] == pytest.approx(published, abs=5e-4)
    assert result["lower"] < result["estimate"] < result["upper"]


def test_rejects_a_run_without_the_plain_ar_arm():
    """The archived runs have no ``native_ar``, and must not be silently used."""
    module = _module()
    with pytest.raises(ValueError, match="missing method native_ar"):
        module.build_cell("dflash", "8b", RUN)
