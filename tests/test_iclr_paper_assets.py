from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = ROOT / "scripts" / "build_iclr_paper_assets.py"
    spec = importlib.util.spec_from_file_location("build_iclr_paper_assets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loads_complete_recorded_evidence() -> None:
    builder = _load_builder()
    data = builder.load_paper_data(ROOT)

    assert len(data["main_math"]) == 4
    assert len(data["breadth_rows"]) == 16
    assert {row["requests"] for row in data["breadth_rows"]} == {128, 160, 164, 378}
    assert data["breadth_aggregate"]["positive_speed_ci_cells"] == 11
    assert data["breadth_aggregate"]["amdahl_direction_matches"] == 16
    assert (
        abs(
            data["breadth_aggregate"]["raw_geometric_mean_speedup"] - 1.1135385935857058
        )
        < 1e-12
    )
    assert (
        abs(
            data["breadth_aggregate"]["profile_policy_geometric_mean_speedup"]
            - 1.1354226107888445
        )
        < 1e-12
    )
    assert len(data["heldout_forecasts"]) == 4
    assert (
        abs(data["forecast_mean_absolute_relative_error_percent"] - 1.4471776194524721)
        < 1e-12
    )
    assert data["similarity"]["by_benchmark"]["math500"]["counts"]["0.80"] == 47
    assert data["similarity"]["by_benchmark"]["humaneval"]["maximum"] < 0.30
    assert all(row["ideal_speedup"] >= row["speedup"] for row in data["main_math"])
    assert all(
        abs(
            row["acceptance_margin"]
            - (row["acceptance_retention"] - row["break_even_retention"])
        )
        < 1e-12
        for row in data["breadth_rows"]
    )


def test_main_math_values_match_final_artifacts() -> None:
    builder = _load_builder()
    rows = {
        (row["family"], row["target"]): row
        for row in builder.load_paper_data(ROOT)["main_math"]
    }

    assert abs(rows[("DFlash", "8B")]["speedup"] - 1.4830972238360247) < 1e-12
    assert abs(rows[("DFlash", "14B")]["speedup"] - 1.245845862573604) < 1e-12
    assert abs(rows[("EAGLE-3", "8B")]["speedup"] - 1.2480383920085085) < 1e-12
    assert abs(rows[("EAGLE-3", "14B")]["speedup"] - 1.0815729706441708) < 1e-12
    assert rows[("DFlash", "8B")]["requests"] == 500
    assert rows[("EAGLE-3", "14B")]["exact_matches"] == 498


def test_generated_assets_are_reproducible(tmp_path: Path) -> None:
    builder = _load_builder()
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.build_all(ROOT, first)
    builder.build_all(ROOT, second)

    first_files = sorted(
        path.relative_to(first) for path in first.rglob("*") if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second) for path in second.rglob("*") if path.is_file()
    )
    assert first_files == second_files
    assert first_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()

    expected_figures = {
        Path("figures/system_overview.pdf"),
        Path("figures/main_throughput.pdf"),
        Path("figures/mechanism.pdf"),
        Path("figures/breadth_and_margin.pdf"),
        Path("figures/acceptance_survival.pdf"),
        Path("figures/memory_reduction.pdf"),
    }
    assert expected_figures <= set(first_files)
