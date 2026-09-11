import json
import pytest
from experiments.handoff_transfer.collect_scaling import collect


def test_scaling_collection_rejects_different_runtime(tmp_path):
    root = tmp_path / 'run'
    baseline = tmp_path / 'baseline'
    (root / 'measurements').mkdir(parents=True)
    baseline.mkdir()
    (root / 'transfer.json').write_text(json.dumps({'family': 'q8', 'records': 16,
        'initialization_training_seconds': 2}))
    for mode in ('ar', 'zip', 'handoff-r56', 'handoff-five'):
        for repeat in range(3):
            path = (baseline if mode == 'ar' else root / 'measurements') / f'{mode}-r{repeat}-w0.jsonl'
            path.write_text('\n'.join(json.dumps({'group_id': str(i), 'timing_valid': True,
                'output_ids': [1], 'prompt_ids': [2], 'output_tokens': 1,
                'wall_seconds': 2 if mode == 'ar' else 1, 'finish_reason': 'stop'}) for i in range(128)))
            path.with_suffix('.summary.json').write_text(json.dumps({'timing_valid': True,
                'contract': {'count': 128, 'cap': 2048, 'repeat': repeat, 'family': 'q8',
                             'runtime_config': {'max_model_len': 5120}, 'manifest_sha256': 'same'}}))
    result = collect(root, baseline)
    assert result['all_exact'] and result['rows']['handoff-r56']['mean_speedup_over_ar'] == 2
    path = root / 'measurements/handoff-five-r2-w0.summary.json'
    bad = json.loads(path.read_text())
    bad['contract']['runtime_config']['max_model_len'] = 4096
    path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError, match='Runtime'):
        collect(root, baseline)
