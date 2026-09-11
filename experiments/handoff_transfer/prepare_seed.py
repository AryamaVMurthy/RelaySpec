"""Prepare a seed-specific Qwen8 or Llama transfer using a freshly fitted ZIP initializer."""
import argparse
import json
from pathlib import Path
from experiments.handoff_transfer.prepare_scaling import prepare
from experiments.handoff_transfer.seed_source import seed_training_source
from experiments.handoff_transfer.initializer_contract import initializer_records
from experiments.handoff_transfer.pack_transfer import sha, write


def main(a):
    summary = json.loads((a.fit / 'summary.json').read_text())
    parent = json.loads((a.parent / 'transfer.json').read_text())
    assert parent['family'] in ('q8', 'llama')
    config, _ = initializer_records(summary, parent['family'], 4096)
    assert config['seed'] == a.seed
    prepare(a.parent, a.fit, a.out, 4096)
    transfer = json.loads((a.out / 'transfer.json').read_text())
    transfer.update(initialization_seed=a.seed, training_seed=a.seed,
                    seed_scope='ZIP initializer and AUF initialization, anchor sampling, shard shuffle; original frozen targets and cached rollouts shared.')
    write(a.out / 'transfer.json', transfer)
    source = Path(__file__).with_name('port') / 'train.py'
    (a.out / 'seed_train.py').write_text(seed_training_source(source.read_text(), a.seed))
    write(a.out / 'seed-provenance.json', {'seed': a.seed, 'family': parent['family'],
        'initializer_summary_sha256': sha(a.fit / 'summary.json'),
        'trainer_sha256': sha(a.out / 'seed_train.py'),
        'normal_baseline': 'Must also run train_normal --seed with this seed; not yet trained by preparation.',
        'main_seed42_source_sha256': sha(source), 'status': 'prepared, not trained'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'fit', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(43, 44), required=True)
    main(p.parse_args())
