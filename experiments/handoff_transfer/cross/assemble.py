"""Index full cross-family data without loading hundreds of GB of feature tensors."""
import argparse
import json
import statistics
from pathlib import Path
from experiments.handoff_transfer.matrix.prepare import digest


def anchor_coverage(counts, limit=32):
    assert counts and all(type(n) is int and n>0 for n in counts)
    return {'records':len(counts),'anchor_limit':limit,
            'minimum_eligible':min(counts),'median_eligible':statistics.median(counts),
            'mean_eligible':statistics.mean(counts),'maximum_eligible':max(counts),
            'records_below_limit':sum(n<limit for n in counts),
            'distinct_anchors_available_per_complete_epoch':sum(min(n,limit) for n in counts),
            'scope':'Availability in each record, not actual anchors consumed by a shuffled partial-epoch fit'}


def assemble(chunks, records=4096):
    index, seen, manifest_hashes = [], set(), set()
    for chunk in chunks:
        rows = json.loads((chunk / 'train.json').read_text())
        generation = json.loads((chunk / 'train-generation.json').read_text())
        alignment = json.loads((chunk / 'source-label-alignment.json').read_text())
        labels = json.loads((chunk / 'alignment-summary.json').read_text())
        conditioning = json.loads((chunk / 'conditioning-contract.json').read_text())
        assert generation['teacher'] == 'unsloth/Llama-3.1-8B-Instruct'
        assert generation['target_adapters'] is None and generation['output_cap'] == 4096
        assert generation['temperature'] == 0 and generation['seed'] == 42
        assert generation['offset'] == len(index), 'Noncontiguous rollout chunks'
        assert generation['records'] == len(rows) == len(alignment)
        rollout_hash = digest(chunk / 'train.json')
        assert generation['sha256'] == labels['rollout_sha256'] == conditioning['train_sha256'] == rollout_hash
        assert labels['labels_sha256'] == digest(chunk / 'source-label-alignment.json')
        assert conditioning['causality_check']['passed']
        assert conditioning['target_taps'] == [1, 8, 15, 22, 29]
        manifest_hashes.add(generation['source_manifest_sha256'])
        for i, (row, aligned) in enumerate(zip(rows, alignment)):
            key = row['group_id']
            assert key == aligned['group_id'] and key not in seen
            assert aligned['blocks'], 'No aligned source labels'
            assert row['full_ids'] == row['prompt_token_ids'] + row['output_ids']
            assert 0 < len(row['output_ids']) <= 4096
            feature = chunk / f'features/source/train/{i:05d}.pt'
            provenance = json.loads(feature.with_suffix('.json').read_text())
            assert provenance['manifest_sha256'] == rollout_hash and feature.is_file()
            # No tensor import or full feature scan here. The training loader must
            # verify the retained hash before first use of each tensor file.
            index.append({'group_id': key, 'chunk': str(chunk), 'row_index': i,
                          'feature_path': str(feature), 'feature_sha256': provenance['sha256'],
                          'rollout_sha256': rollout_hash, 'alignment_sha256': labels['labels_sha256'],
                          'eligible_anchors': len(aligned['blocks'])})
            seen.add(key)
    assert len(index) == records and len(manifest_hashes) == 1
    return {'records': records, 'source_manifest_sha256': next(iter(manifest_hashes)),
            'anchor_coverage':anchor_coverage([r['eligible_anchors'] for r in index]),
            'scope': 'cross-family full-data index; not a fitted model or benchmark',
            'feature_hash_validation': 'required on first use by trainer', 'index': index}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--chunks', type=Path, nargs='+', required=True)
    p.add_argument('--records', type=int, default=4096)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = assemble(a.chunks, a.records)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as handle:
        json.dump(result, handle, indent=2)
