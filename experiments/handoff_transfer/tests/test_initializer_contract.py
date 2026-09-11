import hashlib
import json
import pytest
from experiments.handoff_transfer.initializer_contract import initializer_records


def test_family_initializer_checks_numeric_order_and_manifest_hashes(tmp_path):
    manifests = {}
    for part, group in ((10, 'second'), (2, 'first')):
        p = tmp_path / f'part-{part}/train.json'; p.parent.mkdir()
        p.write_text(json.dumps([{'group_id': group}]))
        manifests[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    summary = {'status': 'fit_complete_offline_validation_pending',
        'history': [{'epoch': i} for i in (1, 2, 3)], 'contract': {'family': 'llama',
        'objective': 'zip', 'target_adapters': None, 'records': 2, 'epochs': 3,
        'data': str(tmp_path), 'manifests': manifests, 'seed': 43}}
    config, rows = initializer_records(summary, 'llama', 2)
    assert config['seed'] == 43 and [r['group_id'] for r in rows] == ['first', 'second']
    (tmp_path / 'part-2/train.json').write_text('[{"group_id":"changed"}]')
    with pytest.raises(AssertionError):
        initializer_records(summary, 'llama', 2)
