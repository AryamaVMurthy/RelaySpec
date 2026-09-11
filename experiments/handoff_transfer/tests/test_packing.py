import argparse
import json
import torch
import pytest
from experiments.handoff_transfer.pack_transfer import main, sha, write


def test_variable_count_packing_checks_capture_identity(tmp_path):
    data = tmp_path / 'main-q8-n32'
    rows = [{'group_id': str(i), 'full_ids': [i], 'prompt_token_ids': [], 'output_ids': [i]} for i in range(32)]
    manifest = data / 'train.json'
    write(manifest, rows)
    for i in range(32):
        path = data / f'features/8/train/{i:05d}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'group_id': str(i), 'features': torch.zeros(1, 20480, dtype=torch.bfloat16)}, path)
        write(path.with_suffix('.json'), {'manifest_sha256': sha(manifest), 'sha256': sha(path)})
    export = tmp_path / 'fits/q8-n32-zip-lr1e-3-s42/epoch-3/export'
    export.mkdir(parents=True)
    (export / 'model.safetensors').write_bytes(b'fixture')
    args = argparse.Namespace(records=32, family='q8', data_name=None, study=tmp_path,
                              out=tmp_path / 'packed', base_export=None)
    main(args)
    assert json.loads((args.out / 'transfer.json').read_text())['records'] == 32
    assert len(torch.load(args.out / 'features/full/00000.pt', weights_only=True)['rows']) == 32
    # A corrupted capture must not be accepted in a new packing destination.
    path = data / 'features/8/train/00000.json'
    meta = json.loads(path.read_text())
    meta['manifest_sha256'] = 'wrong'
    write(path, meta)
    args.out = tmp_path / 'rejected'
    with pytest.raises(AssertionError):
        main(args)
