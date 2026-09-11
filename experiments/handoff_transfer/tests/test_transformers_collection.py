import json
import pytest
from experiments.handoff_transfer.collect_transformers import collect


def test_transformers_collection_rejects_checkpoint_mixup(tmp_path):
    primary = tmp_path / 'primary'; primary.mkdir()
    for phase, count, cap in (('gate', 4, 128), ('full', 128, 2048)):
        folder = tmp_path / phase; folder.mkdir()
        for mode in ('ar', 'normal', 'zip', 'handoff-r56', 'handoff-five'):
            contract = {'family': 'q8', 'mode': mode, 'count': count, 'cap': cap, 'repeat': 0,
                        'export_sha256': None if mode == 'ar' else mode, 'manifest_sha256': 'same',
                        'runtime_config': {'backend': 'transformers'}}
            path = folder / f'{mode}-r0-w0.jsonl'
            path.with_suffix('.summary.json').write_text(json.dumps({'timing_valid': True, 'contract': contract}))
            path.write_text('\n'.join(json.dumps({'group_id': i, 'prompt_ids': [1], 'output_ids': [2],
                'finish_reason': 'stop', 'output_tokens': 1, 'wall_seconds': 2 if mode == 'ar' else 1,
                'timing_valid': True}) for i in range(count)))
            (primary / f'{mode}-r0-w0.summary.json').write_text(json.dumps({'contract': contract}))
    report = collect(tmp_path, primary, 'q8')
    assert report['all_exact'] and report['phases']['full']['handoff-r56']['throughput_ratio'] == 2
    path = primary / 'zip-r0-w0.summary.json'
    metadata = json.loads(path.read_text()); metadata['contract']['export_sha256'] = 'wrong'
    path.write_text(json.dumps(metadata))
    with pytest.raises(AssertionError, match='another checkpoint'):
        collect(tmp_path, primary, 'q8')
