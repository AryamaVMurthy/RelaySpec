"""Validate persisted feature files before recovering a failed capture job."""
import argparse
import json
from pathlib import Path
from .pilot_data import sha, write


def main(args):
    import torch
    width = 5120 if args.role == 'target' else 2560
    taps = [1, 10, 19, 28, 37] if args.role == 'target' else [1, 9, 17, 25, 33]
    manifest = args.out / 'train.json'
    rows = json.loads(manifest.read_text())
    assert len(rows) == 512
    manifest_hash = sha(manifest)
    base = args.out / 'features' / args.role / 'train'
    assert json.loads((base / 'causality.json').read_text())['passed']
    for index, row in enumerate(rows):
        path = base / f'{index:05d}.pt'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['role'] == args.role
        assert meta['manifest_sha256'] == manifest_hash
        assert meta['sha256'] == sha(path)
        data = torch.load(path, map_location='cpu', weights_only=True)
        assert data['group_id'] == row['group_id'] and data['layers'] == taps
        assert data['features'].shape == (len(row['full_ids']), 5 * width)
        assert data['features'].dtype == torch.bfloat16
        assert torch.isfinite(data['features']).all()
        del data
    report = {'records': len(rows), 'role': args.role, 'manifest_sha256': manifest_hash,
              'checks': 'SHA256, manifest identity, group, taps, shape, BF16, finite, causality'}
    write(args.out / f'{args.role}-recovery-verification.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--role', choices=['target', 'source'], required=True)
    main(p.parse_args())
