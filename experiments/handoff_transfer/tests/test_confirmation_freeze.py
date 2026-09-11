import hashlib
import json
import pytest
from experiments.handoff_transfer.freeze_confirmation import freeze


def test_confirmation_freeze_rejects_changed_checkpoint_and_overwrite(tmp_path):
    reports, workloads, root = [tmp_path / x for x in ('reports', 'workloads', 'runs')]
    reports.mkdir(); workloads.mkdir()
    valid = {'status': 'complete', 'family': 'q14', 'all_exact': True, 'timing_repetitions': 3}
    (reports / 'q14-comparison.json').write_text(json.dumps(valid))
    for workload in ('math', 'gsm', 'code', 'chat'):
        (workloads / f'q14-{workload}.json').write_text(json.dumps(valid))
    folder = root / 'q14/measurements'; folder.mkdir(parents=True)
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    (checkpoint / 'model.safetensors').write_bytes(b'weights')
    (checkpoint / 'config.json').write_text('{}')
    weight_hash = hashlib.sha256(b'weights').hexdigest()
    for mode in ('ar', 'normal', 'zip', 'handoff-r56', 'handoff-five'):
        runtime = {} if mode == 'ar' else {'speculative_config': {'model': str(checkpoint)}}
        (folder / f'{mode}-r0-w0.summary.json').write_text(json.dumps({'timing_valid': True,
            'contract': {'family': 'q14', 'mode': mode, 'count': 128, 'cap': 2048,
                         'runtime_config': runtime, 'export_sha256': weight_hash}}))
    output = tmp_path / 'frozen.json'
    assert freeze(root, reports, workloads, output, ['q14'])['status'] == 'frozen'
    with pytest.raises(AssertionError, match='overwrite'):
        freeze(root, reports, workloads, output, ['q14'])
    (checkpoint / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(AssertionError, match='Checkpoint changed'):
        freeze(root, reports, workloads, tmp_path / 'other.json', ['q14'])
    assert not (tmp_path / 'other.json').exists()
