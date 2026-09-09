"""Audit complete Spark replication and summarize sampled metrics, never invent N/A."""
import argparse
import json
import math
from pathlib import Path


def numeric(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def energy(samples, start, stop):
    """Trapezoidal integration with interpolated endpoints; require full coverage."""
    import numpy as np
    points = [(r["monotonic"], numeric(r["devices"][0].get("power.draw")) if len(r["devices"]) == 1 else None)
              for r in samples]
    # Missing samples inside an interval must not silently disappear.
    left = [p for p in points if p[0] <= start]
    right = [p for p in points if p[0] >= stop]
    if not left or not right:
        return None
    selected = [left[-1]] + [p for p in points if start < p[0] < stop] + [right[0]]
    if any(p[1] is None for p in selected) or any(b[0]-a[0] > 3 for a,b in zip(selected,selected[1:])):
        return None
    ts, ps = zip(*selected)
    x = [start] + [t for t in ts if start < t < stop] + [stop]
    y = np.interp(x, ts, ps)
    return float(sum((b-a)*(u+v)/2 for a,b,u,v in zip(x,x[1:],y,y[1:])))


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    done = json.loads((out/"complete.json").read_text()); assert done["status"] == "pass"
    cfg = json.loads((out/"config.json").read_text())
    rows = [json.loads(s) for s in (out/"evaluation.jsonl").read_text().splitlines()]
    samples = [json.loads(s) for s in (out/"telemetry.jsonl").read_text().splitlines()]
    methods = ["native", "compact_linear", "candidate", "reference"]
    ids = sorted({r["problem_id"] for r in rows})
    lookup = {(r["method"],r["problem_id"],r["repeat"]): r for r in rows}
    assert len(rows) == len(lookup) == cfg["requests"]*4*cfg["repeats"]
    assert len(ids) == cfg["requests"]
    for m in methods:
        for p in ids:
            for t in range(cfg["repeats"]):
                a,b = lookup[m,p,0],lookup[m,p,t]
                assert a["tokens"] == b["tokens"] and a["acceptance_lengths"] == b["acceptance_lengths"]
                assert 0 < b["seconds"] and len(b["tokens"]) == b["output_tokens"] <= cfg["output_cap"]
    ar = {r["problem_id"]: r for r in map(json.loads,(out/"ar-quality.jsonl").read_text().splitlines())}
    assert set(ar) == set(ids)
    arrays = {m:np.array([[sum(lookup[m,p,t][k] for t in range(cfg["repeats"])) for k in ["output_tokens","seconds"]] for p in ids]) for m in methods}
    rng = np.random.default_rng(1729)
    strata = [[i for i,p in enumerate(ids) if lookup["native",p,0]["benchmark"] == b] for b in ["gsm8k","math500","humaneval","mtbench"]]
    draws = np.concatenate([rng.choice(s,(10000,len(s)),replace=True) for s in strata],axis=1)
    aggregate = []
    for m in methods:
        arm = [r for r in rows if r["method"] == m]
        total = arrays[m].sum(0)
        ratios = {}
        for reference in ["native", "reference"]:
            a,b = arrays[m][draws].sum(1),arrays[reference][draws].sum(1)
            ratios[reference] = {"ratio":float((total[0]/total[1])/(arrays[reference].sum(0)[0]/arrays[reference].sum(0)[1])),
                                 "ci95":np.quantile((a[:,0]/a[:,1])/(b[:,0]/b[:,1]),[.025,.975]).tolist()}
        metrics = {}
        selected = [s for s in samples if any(r["monotonic_start"] <= s["monotonic"] <= r["monotonic_stop"] for r in arm)]
        for key in ["utilization.gpu","utilization.memory","power.draw","temperature.gpu","clocks.current.sm","clocks.current.memory","memory.used"]:
            values = [numeric(s["devices"][0].get(key)) for s in selected if len(s["devices"]) == 1]
            valid = [x for x in values if x is not None]
            metrics[key] = {"samples":len(valid),"missing_samples":len(selected)-len(valid),
                "mean":float(np.mean(valid)) if valid else None,"max":max(valid) if valid else None,
                "p95":float(np.quantile(valid,.95)) if valid else None}
        energies = [energy(samples,r["monotonic_start"],r["monotonic_stop"]) for r in arm]
        joules = sum(energies) if all(e is not None for e in energies) else None
        aggregate.append({"method":m,"tps":float(total[0]/total[1]),"ratios":ratios,
            "exact_native":sum(lookup[m,p,0]["tokens"] == lookup["native",p,0]["tokens"] for p in ids),
            "exact_ar":sum(lookup[m,p,0]["tokens"] == ar[p]["tokens"] for p in ids),
            "capped":sum(lookup[m,p,0]["capped"] for p in ids),
            "progress":sum(sum(r["acceptance_lengths"]) for r in arm)/sum(len(r["acceptance_lengths"]) for r in arm),
            "max_peak_allocated_bytes":max(r["peak_allocated"] for r in arm),
            "max_peak_reserved_bytes":max(r["peak_reserved"] for r in arm),
            "max_incremental_allocated_bytes":max(r["incremental_peak_allocated"] for r in arm),
            "min_system_available_bytes":min((s["system_memory"]["MemAvailable_bytes"] for s in selected),default=None),
            "max_process_rss_bytes":max((s["process_memory"].get("VmRSS_bytes",0) for s in selected),default=None),
            "estimated_device_joules":joules,"estimated_device_joules_per_token":joules/total[0] if joules is not None else None,
            "metrics":metrics})
    summary={"status":"complete","requests":len(ids),"output_cap":cfg["output_cap"],"repeats":cfg["repeats"],"rows":aggregate,
        "scope":"Same four fixed methods on previously evaluated prompts. Sampled power is device-reported, not wall-socket power; energy is approximate. GPU memory N/A is preserved; UMA system memory and Torch allocations are separate. All models resident; memory is not isolated deployment. Confidence intervals exclude seed and hardware/runtime uncertainty."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+'\n')
    lines=["# DGX Spark hardware replication","",summary["scope"],"","| Method | TPS | / Original | / Full DDTree | Exact AR | Peak Torch allocated GiB | Mean device W |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        power=r["metrics"]["power.draw"]["mean"]
        lines.append(f"| {r['method']} | {r['tps']:.1f} | {r['ratios']['native']['ratio']:.3f} | {r['ratios']['reference']['ratio']:.3f} | {r['exact_ar']}/{len(ids)} | {r['max_peak_allocated_bytes']/2**30:.2f} | {f'{power:.1f}' if power is not None else 'N/A'} |")
    (out/"SUMMARY.md").write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
