import json
import pytest
from experiments.handoff_transfer.collect_profiles import collect, MODES


def test_profile_collection_rejects_primary_timing(tmp_path):
    for mode in MODES:
        root = tmp_path / f'{mode}-123'
        (root / 'requests').mkdir(parents=True)
        (root / 'traces').mkdir()
        summary = root / f'requests/{mode}-r0-w0.summary.json'
        summary.write_text(json.dumps({'timing_valid': False, 'contract': {'family': 'q8',
            'mode': mode, 'count': 4, 'cap': 2048, 'manifest_sha256': 'same',
            'runtime_config': {'profiler_config': {'max_iterations': 64}}}}))
        summary.with_name(f'{mode}-r0-w0.jsonl').write_text('\n'.join(json.dumps({'group_id': i,
            'timing_valid': False}) for i in range(4)))
        (root / 'traces/rank0.pt.trace.json').write_text(json.dumps({'traceEvents': [
            {'cat': 'kernel', 'ph': 'X', 'name': 'gemm', 'ts': 0, 'dur': 100}]}))
        (root / 'gpu.csv').write_text('timestamp,index\nnow,0\n')
    result = collect(tmp_path, 'q8', '123')
    assert len(result['methods']) == 5
    assert result['methods']['ar']['traces'][0]['kernel_union_ms'] == .1
    summary = tmp_path / 'ar-123/requests/ar-r0-w0.summary.json'
    data = json.loads(summary.read_text()); data['timing_valid'] = True
    summary.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match='primary timing'):
        collect(tmp_path, 'q8', '123')
