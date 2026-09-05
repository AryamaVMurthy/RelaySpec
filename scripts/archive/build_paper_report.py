from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run label and output directory as LABEL=PATH.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def interval(value: Any, digits: int = 3) -> str:
    if not value:
        return "—"
    return f"[{float(value[0]):.{digits}f}, {float(value[1]):.{digits}f}]"


def load_run(specification: str) -> tuple[str, Path, dict[str, Any], dict[str, Any]]:
    if "=" not in specification:
        raise ValueError("run must be specified as LABEL=PATH")
    label, raw_path = specification.split("=", 1)
    path = Path(raw_path).resolve()
    summary = json.loads(
        (path / "benchmark-paper-summary.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    return label, path, summary, config


def source_fraction(method: dict[str, Any]) -> float | None:
    shares = method.get("profile_region_request_shares")
    if not shares:
        return None
    return sum(float(value) for key, value in shares.items() if "source_trunk" in key)


def relay_fraction(method: dict[str, Any]) -> float | None:
    shares = method.get("profile_region_request_shares")
    if not shares:
        return None
    return sum(
        float(value)
        for key, value in shares.items()
        if key in {"relay", "prefill_relay"}
    )


def main() -> None:
    args = parse_args()
    runs = [load_run(value) for value in args.run]
    lines = [
        "# RelaySpec benchmark report",
        "",
        "All speed intervals are prompt-paired 95% bootstrap intervals.",
        "",
        "## Main results",
        "",
        "| Run | Benchmark | Method | Accuracy | Δ accuracy vs source | End-to-end tok/s | End-to-end speedup vs source | 95% CI | Acceptance | Cap rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _, summary, _ in runs:
        for benchmark, benchmark_summary in summary["by_benchmark"].items():
            for method_name, method in benchmark_summary["methods"].items():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            label,
                            benchmark,
                            method_name,
                            metric(method.get("accuracy")),
                            metric(method.get("accuracy_delta_vs_source_reuse")),
                            metric(method.get("end_to_end_tokens_per_second"), 1),
                            metric(method.get("end_to_end_speedup_vs_source_reuse")),
                            interval(
                                method.get(
                                    "end_to_end_speedup_vs_source_reuse_ci95"
                                )
                            ),
                            metric(method.get("mean_acceptance_length")),
                            metric(method.get("cap_hit_rate")),
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Amdahl and component evidence",
            "",
            "| Run | Method | Source-trunk wall share | Relay wall share | Profile accounted |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for label, _, summary, _ in runs:
        for method_name, method in summary["methods"].items():
            lines.append(
                f"| {label} | {method_name} | {metric(source_fraction(method))} | "
                f"{metric(relay_fraction(method))} | "
                f"{metric(method.get('profile_accounted_fraction'))} |"
            )
    lines.extend(["", "## Reproducibility", ""])
    for label, path, summary, config in runs:
        lines.extend(
            [
                f"- `{label}`: `{path}`",
                f"  - target: `{config['target']['id']}` @ `{config['target']['revision']}`",
                f"  - scorer: `{summary.get('scorer', 'unknown')}`",
                f"  - source hashes: `{path / 'source-sha256.txt'}`",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
