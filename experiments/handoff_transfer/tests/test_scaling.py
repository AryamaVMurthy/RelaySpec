import json

import pytest
import torch

from experiments.handoff_transfer.pack_transfer import sha, write
from experiments.handoff_transfer.prepare_scaling import prepare


def test_scaling_rejects_extra_initialization_data_and_truncates_capture(tmp_path):
    parent, fit, out = [tmp_path / name for name in ('parent', 'fit', 'out')]
    rows = [{'group_id': str(i), 'full_ids': [i]} for i in range(32)]
    manifest = tmp_path / 'train.json'
    write(manifest, rows)
    shard = parent / 'features/full/00000.pt'
    shard.parent.mkdir(parents=True)
    torch.save({'rows': rows, 'features': [torch.ones(1, 2) for _ in rows]}, shard)
    write(parent / 'transfer.json', {'status': 'complete', 'shards': {str(shard): sha(shard)}})
    export = fit / 'epoch-3/export'
    export.mkdir(parents=True)
    (export / 'model.safetensors').write_bytes(b'fixture')
    (fit / 'resume.pt').write_bytes(b'fixture')
    summary = {'status': 'complete', 'config': {'records': 32, 'epochs': 3, 'manifest': str(manifest)},
               'checkpoint_sha256': sha(fit / 'resume.pt'),
               'history': [{'epoch': i, 'seconds': 1} for i in (1, 2, 3)]}
    write(fit / 'summary.json', summary)
    with pytest.raises(AssertionError):
        prepare(parent, fit, out, 16)
    assert not out.exists()
    summary['config']['records'] = 16
    write(fit / 'summary.json', summary)
    prepare(parent, fit, out, 16)
    packed = torch.load(out / 'features/full/00000.pt', weights_only=True)
    assert packed['rows'] == rows[:16]
    assert len(packed['features']) == 16
    contract = json.loads((out / 'transfer.json').read_text())
    assert contract['records'] == 16
    assert contract['initialization_training_seconds'] == 3
    trainer = (out / 'scaling_train.py').read_text()
    assert "else TRANSFER['records'];total_steps" in trainer
    compile(trainer, 'scaling_train.py', 'exec')
