"""Collect instrumented kernel evidence independently of primary timing."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.auf_vllm.analyze_gpu_profiles import analyze

MODES = ('ar', 'normal', 'zip', 'handoff-r56', 'handoff-five')


def collect(root, family, job):
    reports = {}
    signatures = set()
    for mode in MODES:
        folder = root / f'{mode}-{job}'
        path = folder / f'requests/{mode}-r0-w0.jsonl'
        summary = path.with_suffix('.summary.json')
        metadata = json.loads(summary.read_text())
        contract = metadata['contract']
        assert metadata['timing_valid'] is False, 'Profiling must never be primary timing'
        assert contract['family'] == family and contract['mode'] == mode
        assert contract['count'] == 4 and contract['cap'] == 2048
        assert contract['runtime_config']['profiler_config']['max_iterations'] == 64
        signatures.add(contract['manifest_sha256'])
        requests = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(requests) == len({r['group_id'] for r in requests}) == 4
        assert all(r['timing_valid'] is False for r in requests)
        traces = sorted(p for p in (folder / 'traces').rglob('*')
                        if p.name.endswith(('.pt.trace.json', '.pt.trace.json.gz')))
        assert traces, f'Missing GPU trace for {mode}'
        analyzed = []
        for trace in traces:
            evidence = analyze(trace)
            evidence['source_path'] = str(trace)
            evidence['scope'] = 'At most 64 profiled engine iterations from four requests capped at 2048; not full-request profiling or SM occupancy.'
            analyzed.append(evidence)
        telemetry = folder / 'gpu.csv'
        assert telemetry.exists() and len(telemetry.read_text().splitlines()) > 1
        reports[mode] = {'traces': analyzed, 'summary': metadata,
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, summary, telemetry)},
            'telemetry_note': 'Raw nvidia-smi samples cover loading, warmup and evaluation; may include other visible GPUs. No attribution or energy estimate is inferred.'}
    assert len(signatures) == 1, 'Different evaluation manifests'
    return {'status': 'complete', 'family': family, 'profile_job': job, 'methods': reports,
            'scope': 'Instrumentation-only evidence. No profiled TPS is reported; use separate unprofiled main measurements for throughput. Kernel categories are name-based heuristics.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--family', choices=('q8', 'llama'), required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); report = collect(a.root, a.family, a.job)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
