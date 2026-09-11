"""Read the audited Qwen8 and family ZIP initializer schemas explicitly."""
import hashlib
import json
from pathlib import Path


def initializer_records(summary, family, records):
    if family == 'q8':
        assert summary['status'] == 'complete'
        config = summary['config']
        rows = json.loads(Path(config['manifest']).read_text())[:records]
    else:
        assert family in ('llama', 'q14')
        assert summary['status'] == 'fit_complete_offline_validation_pending'
        config = summary['contract']
        assert config['family'] == family and config['objective'] == 'zip'
        assert config['target_adapters'] is None
        root = Path(config['data'])
        paths = [root / 'train.json'] if (root / 'train.json').exists() else sorted(
            root.glob('part-*/train.json'), key=lambda p: int(p.parent.name.split('-')[-1]))
        assert {str(p) for p in paths} == set(config['manifests'])
        rows = []
        for path in paths:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == config['manifests'][str(path)]
            rows.extend(json.loads(path.read_text()))
        rows = rows[:records]
    assert config['records'] == records and config['epochs'] == 3
    assert [h['epoch'] for h in summary['history']] == [1, 2, 3]
    assert len(rows) == len({r['group_id'] for r in rows}) == records
    return config, rows
