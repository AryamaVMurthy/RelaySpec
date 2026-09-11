"""Build isolated, matched-data AUF inputs without duplicating dense captures.

The ZIP initializer must already be fitted on exactly the selected records.
This prepares inputs only; it never launches GPU work.
"""
import argparse
import json
from pathlib import Path

import torch

from experiments.handoff_transfer.pack_transfer import sha, write


def training_source():
    source = (Path(__file__).parent / 'port/train.py').read_text()
    for old, new in [
        ('else 4096;total_steps', "else TRANSFER['records'];total_steps"),
        ("'unique_cached_examples':4096", "'unique_cached_examples':count"),
    ]:
        assert source.count(old) == 1
        source = source.replace(old, new)
    return source


def prepare(parent, fit, out, records):
    assert records >= 16 and records % 16 == 0
    assert not out.exists(), 'Use a new output directory; never replace a running study'
    contract = json.loads((parent / 'transfer.json').read_text())
    summary = json.loads((fit / 'summary.json').read_text())
    assert contract['status'] == summary['status'] == 'complete'
    assert summary['config']['records'] == records
    assert summary['config']['epochs'] == 3
    assert [h['epoch'] for h in summary['history']] == [1, 2, 3]
    expected = json.loads(Path(summary['config']['manifest']).read_text())[:records]
    assert len(expected) == records
    assert len({r['group_id'] for r in expected}) == records
    export = fit / 'epoch-3/export'
    assert (export / 'model.safetensors').is_file()
    assert (fit / 'resume.pt').is_file()
    assert summary['checkpoint_sha256'] == sha(fit / 'resume.pt')
    selected = []
    offset = 0
    for path in sorted((parent / 'features/full').glob('*.pt')):
        if offset == records:
            break
        digest = sha(path)
        assert contract['shards'][str(path)] == digest
        shard = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
        take = min(records - offset, len(shard['rows']))
        assert shard['rows'][:take] == expected[offset:offset + take], 'Initializer/capture sequence mismatch'
        selected.append((path, take, len(shard['rows']), digest))
        offset += take
    assert offset == records, 'Dense captures are incomplete for this data size'
    out.mkdir(parents=True)
    shards = {}
    offset = 0
    for path, take, available, digest in selected:
        dest = out / f'features/full/{offset:05d}.pt'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if take == available:
            dest.symlink_to(path.resolve())
        else:
            shard = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
            torch.save({'rows': shard['rows'][:take],
                        'features': [h.clone() for h in shard['features'][:take]]}, dest)
        # A complete shard is the identical verified file through a symlink.
        # Only a newly materialized partial shard needs another hash pass.
        shards[str(dest)] = digest if take == available else sha(dest)
        offset += take
    contract.update(records=records, base_export=str(export),
                    base_sha256=sha(export / 'model.safetensors'), shards=shards,
                    initialization_fit_summary_sha256=sha(fit / 'summary.json'),
                    selected_group_ids=[r['group_id'] for r in expected],
                    initialization_training_seconds=sum(h['seconds'] for h in summary['history']))
    write(out / 'transfer.json', contract)
    # Preserve the audited main recipe; only count-dependent bookkeeping changes.
    (out / 'scaling_train.py').write_text(training_source())
    write(out / 'scaling-provenance.json', {
        'records': records, 'parent': str(parent),
        'trainer_sha256': sha(out / 'scaling_train.py'),
        'main_trainer_sha256': sha(Path(__file__).parent / 'port/train.py'),
        'status': 'prepared; training not launched',
        'note': 'Use port and vendor on PYTHONPATH; AUF_WORKDIR must point here. '
                'Set enough epochs to reach the declared fixed update budget.',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'fit', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--records', type=int, required=True)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.fit.resolve(), args.out.resolve(), args.records)
