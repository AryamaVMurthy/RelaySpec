"""Audit complete paired GPU measurements; no numbers from unfinished runs."""
import argparse
import json
from pathlib import Path
from metrics import energy, numeric


def summarize(out):
    import numpy as np
    done = json.loads((out / "complete.json").read_text())
    assert done["status"] == "pass" and not done["profiling"]
    provenance = json.loads((out / "provenance.json").read_text())
    rows = list(map(json.loads, (out / "evaluation.jsonl").read_text().splitlines()))
    telemetry = list(map(json.loads, (out / "telemetry.jsonl").read_text().splitlines()))
    methods = provenance["methods"]
    lookup = {(r["method"], r["problem_id"], r["repeat"]): r for r in rows}
    ids = sorted({r["problem_id"] for r in rows})
    repeats = done["repeats"]
    assert len(rows) == len(lookup) == len(methods) * len(ids) * repeats == done["rows"]
    for method in methods:
        for pid in ids:
            for repeat in range(repeats):
                r = lookup[method, pid, repeat]
                assert r["tokens"] == lookup[method, pid, 0]["tokens"]
                assert r["acceptance_lengths"] == lookup[method, pid, 0]["acceptance_lengths"]
                assert 0 < r["seconds"] and len(r["tokens"]) == r["output_tokens"] <= provenance["cap"]
    def aggregate(selected_ids, seed):
        arrays = {m: np.array([[sum(lookup[m,p,t][k] for t in range(repeats)) for k in ["output_tokens", "seconds"]] for p in selected_ids]) for m in methods}
        rng = np.random.default_rng(seed)
        groups = [[i for i,p in enumerate(selected_ids) if lookup[methods[0],p,0]["benchmark"] == b] for b in sorted({lookup[methods[0],p,0]["benchmark"] for p in selected_ids})]
        draws = np.concatenate([rng.choice(g, (10000,len(g)), replace=True) for g in groups], axis=1)
        result = {}
        for m in methods:
            arm = [r for r in rows if r["method"] == m and r["problem_id"] in selected_ids]
            tokens, seconds = arrays[m].sum(0)
            ratios = {}
            for reference in methods:
                b = arrays[reference].sum(0)
                draws_a, draws_b = arrays[m][draws].sum(1), arrays[reference][draws].sum(1)
                ratios[reference] = {"ratio": float((tokens/seconds)/(b[0]/b[1])),
                    "ci95": np.quantile((draws_a[:,0]/draws_a[:,1])/(draws_b[:,0]/draws_b[:,1]), [.025,.975]).tolist()}
            samples = [s for s in telemetry if any(r["monotonic_start"] <= s["monotonic"] <= r["monotonic_stop"] for r in arm)]
            metrics = {}
            for key in ["utilization.gpu", "utilization.memory", "power.draw", "temperature.gpu", "clocks.current.sm", "clocks.current.memory", "memory.used"]:
                valid = [numeric(s["devices"][0].get(key)) for s in samples if len(s["devices"]) == 1]
                valid = [v for v in valid if v is not None]
                metrics[key] = {"valid_samples": len(valid), "missing_samples": len(samples)-len(valid),
                    "mean": float(np.mean(valid)) if valid else None, "max": max(valid) if valid else None,
                    "p95": float(np.quantile(valid,.95)) if valid else None}
            energies = [energy(telemetry,r["monotonic_start"],r["monotonic_stop"]) for r in arm]
            joules = sum(energies) if all(x is not None for x in energies) else None
            result[m] = {"requests": len(selected_ids), "repeats": repeats, "tokens": int(tokens), "seconds": float(seconds),
                "tps": float(tokens/seconds), "ratios": ratios, "device_joules": joules,
                "device_joules_per_token": joules/tokens if joules is not None else None,
                "energy_covered_rows": sum(x is not None for x in energies), "gpu_metrics": metrics,
                "peak_allocated_GiB": max(r["peak_allocated"] for r in arm)/2**30,
                "peak_reserved_GiB": max(r["peak_reserved"] for r in arm)/2**30,
                "incremental_peak_GiB": max(r["incremental_peak_allocated"] for r in arm)/2**30,
                "max_process_rss_GiB": max((s["process_memory"].get("VmRSS_bytes",0) for s in samples),default=0)/2**30,
                "min_system_available_GiB": min((s["system_memory"]["MemAvailable_bytes"] for s in samples),default=0)/2**30,
                "mean_progress_per_cycle": sum(sum(r["acceptance_lengths"]) for r in arm)/sum(len(r["acceptance_lengths"]) for r in arm),
                "capped_requests": sum(lookup[m,p,0]["capped"] for p in selected_ids),
                "exact_ar": sum(lookup[m,p,0]["tokens"] == lookup["ar",p,0]["tokens"] for p in selected_ids) if "ar" in methods else None,
                "exact_native": sum(lookup[m,p,0]["tokens"] == lookup["native",p,0]["tokens"] for p in selected_ids) if "native" in methods else None}
        return result
    result = {"status": "audited", "provenance": provenance, "overall": aggregate(ids,1729),
              "by_workload": {b: aggregate([p for p in ids if lookup[methods[0],p,0]["benchmark"] == b],1729)
                              for b in sorted({r["benchmark"] for r in rows})},
              "uncertainty": "Stratified paired-request bootstrap, 10000 resamples; excludes fitting-seed variation. Previously evaluated 32-request subset.",
              "energy_scope": "Trapezoidal integration of sampled device power; not wall energy. Missing coverage remains unavailable."}
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("output", type=Path)
    result = summarize(p.parse_args().output)
    print(json.dumps({m: {k:r[k] for k in ["tps","exact_ar","device_joules_per_token","peak_allocated_GiB"]} for m,r in result["overall"].items()},indent=2))
