import json
import pytest
from experiments.handoff_transfer.collect_compute import collect


def test_compute_collection_checks_schedule_and_finish(tmp_path):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    for kind, mode in (('fusion_r56', 'handoff-r56'), ('five_maps', 'handoff-five')):
        for steps in (500, 1000, 1500, 2000):
            module = tmp_path / 'modules' / f'full-{kind}' / (kind if steps == 2000 else f'seen_{steps * 8}')
            write(module / 'summary.json', {'optimizer_steps': steps, 'processed_examples': steps * 8,
                  'total_schedule_steps': 2000, 'seed': 42})
            write(module / 'verification.json', {'steps': steps, 'status': 'passed',
                  'frozen_non_fc_exact': True, 'folded_relative_mse': 0})
            root = tmp_path if steps == 2000 else tmp_path / 'compute' / f'updates-{steps}'
            for repeat in range(3):
                for base, method in ((root, mode), (tmp_path, 'ar')):
                    path = base / f'measurements/{method}-r{repeat}-w0.jsonl'
                    write(path.with_suffix('.summary.json'), {'timing_valid': True, 'contract': {
                        'family': 'q8', 'mode': method, 'repeat': repeat, 'count': 128,
                        'cap': 2048, 'manifest_sha256': 'same', 'runtime_config': {'max_model_len': 5120}}})
                    path.write_text('\n'.join(json.dumps({'group_id': str(i), 'timing_valid': True,
                        'output_ids': [1], 'prompt_ids': [2], 'output_tokens': 1,
                        'wall_seconds': 2 if method == 'ar' else 1, 'finish_reason': 'stop'}) for i in range(128)))
    result = collect(tmp_path, 'q8')
    assert result['all_exact'] and len(result['cells']) == 8
    assert result['cells'][0]['mean_speedup_over_ar'] == 2
    path = tmp_path / 'compute/updates-500/measurements/handoff-five-r2-w0.jsonl'
    path.write_text(path.read_text().replace('"stop"', '"length"'))
    assert not collect(tmp_path, 'q8')['all_exact']
    summary = tmp_path / 'modules/full-five_maps/seen_4000/summary.json'
    data = json.loads(summary.read_text()); data['total_schedule_steps'] = 500
    write(summary, data)
    with pytest.raises(AssertionError):
        collect(tmp_path, 'q8')
