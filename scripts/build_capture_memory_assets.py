"""Audit paired selective-capture memory diagnostics, without sampling CIs."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", type=int, default=[28520, 28521, 28522])
    args = parser.parse_args()
    root = Path("reports/autoresearch-20260907")
    ledger = {j["id"]: j for j in json.loads((root / "jobs.json").read_text())["jobs"]}
    results, hashes = [], {}
    for job in args.jobs:
        wave = ledger[job]["wave"]
        specs = json.loads(Path(f"configs/autoresearch/20260907/wave{wave}.json").read_text())["lanes"]
        for lane, spec in enumerate(specs):
            folder = root / f"run-{job}"
            assert json.loads((folder / f"lane{lane}-status.json").read_text())["status"] == "pass"
            report = json.loads((folder / f"lane{lane}/research-result.json").read_text())
            assert report["status"] == report["duplicate_control"]["status"] == "pass"
            path = folder / f"lane{lane}/benchmark-rank0.jsonl"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            grouped = {m: {(r["problem_id"], r["repetition"]): r for r in rows if r["method"] == m}
                       for m in ("relay_base", "relay_selected")}
            a, b = grouped["relay_base"], grouped["relay_selected"]
            assert a.keys() == b.keys() and len(a) == 2 and len(rows) == 4
            expected_hash = spec["checkpoint_sha256"][spec["checkpoint"]]
            for key in a:
                for field in ("output_hash", "output_tokens", "acceptance_lengths", "target_calls", "draft_calls", "input_tokens"):
                    assert a[key][field] == b[key][field], (job, lane, key, field)
                assert a[key]["mapper_checkpoint_sha256"] == b[key]["mapper_checkpoint_sha256"] == expected_hash
            peaks = {m: [r["peak_allocated_memory_bytes"] / 2**30 for r in data.values()]
                     for m, data in grouped.items()}
            inputs = {r["input_tokens"] for r in rows}
            assert len(inputs) == 1
            results.append(dict(job=job, wave=wave, lane=lane, taps=5 if lane % 2 == 0 else 2,
                                input_tokens=inputs.pop(), peak_allocated_gib=peaks,
                                saved_gib=[a[k]["peak_allocated_memory_bytes"] / 2**30 -
                                           b[k]["peak_allocated_memory_bytes"] / 2**30 for k in a],
                                paired_request_time_ratio=[a[k]["request_seconds"] / b[k]["request_seconds"] for k in a],
                                prefill_lifetime_control=wave == 32,
                                exact_trajectory_pairs=len(a)))
    output = dict(input_sha256=hashes, results=results,
                  scope="One synthetic prompt per length, two reversed-order repetitions; descriptive memory diagnostic, not independent samples. Peak allocated includes model weights and both resident mapper copies. Not reserved memory or a task-quality result.")
    (root / "capture-memory-summary.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"Input tokens & Taps & Release prefill & All states (GiB) & Selected (GiB) & Saved (GiB) \\", r"\midrule"]
    for r in results:
        baseline = sum(r["peak_allocated_gib"]["relay_base"]) / 2
        selected = sum(r["peak_allocated_gib"]["relay_selected"]) / 2
        lines.append(f"{r['input_tokens']:,} & {r['taps']} & {'Yes' if r['prefill_lifetime_control'] else 'No'} & {baseline:.2f} & {selected:.2f} & {baseline-selected:.2f} " + r"\\")
        print(r["job"], r["lane"], r["input_tokens"], round(baseline-selected, 3), "GiB saved")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("paper/iclr2027/generated/capture_memory_table.tex").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.5), layout="constrained")
    for taps, color, marker in [(5, "#285a8c", "o"), (2, "#b45520", "s")]:
        points = sorted((r for r in results if r["taps"] == taps and not r["prefill_lifetime_control"]),
                        key=lambda r: r["input_tokens"])
        for method, style, label in [("relay_base", "--", "All states"),
                                     ("relay_selected", "-", "Selected states")]:
            ax.plot([r["input_tokens"] for r in points],
                    [sum(r["peak_allocated_gib"][method]) / 2 for r in points],
                    color=color, marker=marker, linestyle=style, label=f"{taps} taps: {label}")
    ax.set(xlabel="Input tokens (one synthetic archive per length)",
           ylabel="Peak allocated GPU memory (GiB)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(root / f"capture-memory-scaling.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
