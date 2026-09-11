"""Aggregate all declared rank points, retaining measured costs and exactness."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from experiments.auf_vllm.compare_outputs import compare


def collect(root):
    baseline = root / 'q8/measurements'
    signatures, cells = set(), []
    for rank in (8, 16, 32, 56, 128, 256):
        run = root / ('q8' if rank == 56 else f'q8-capacity-r{rank}')
        kind = 'fusion_r56' if rank == 56 else 'fusion_capacity'
        mode = 'handoff-r56' if rank == 56 else 'handoff-capacity'
        model_contract = json.loads((run / f'model-contract-{kind}.json').read_text())
        training = json.loads((run / f'modules/full-{kind}/{kind}/summary.json').read_text())
        verified = json.loads((run / f'modules/full-{kind}/{kind}/verification.json').read_text())
        assert model_contract['rank'] == model_contract['alpha'] == rank
        assert training['optimizer_steps'] == verified['steps'] == 2000
        assert verified['status'] == 'passed'
        repeats, hashes = [], {}
        for repeat in range(3):
            candidate = run / f'measurements/{mode}-r{repeat}-w0.jsonl'
            reference = baseline / f'handoff-r56-r{repeat}-w0.jsonl'
            ar = baseline / f'ar-r{repeat}-w0.jsonl'
            for path in (candidate, reference, ar):
                summary = path.with_suffix('.summary.json')
                contract = json.loads(summary.read_text())['contract']
                assert contract['count'] == 128 and contract['cap'] == 2048
                assert contract['family'] == 'q8' and contract['repeat'] == repeat
                runtime = dict(contract['runtime_config']); runtime.pop('speculative_config', None)
                signatures.add((contract['manifest_sha256'], json.dumps(runtime, sort_keys=True)))
                for source in (path, summary):
                    hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            measured = compare([ar], [candidate])
            matched = compare([reference], [candidate])
            assert measured['count'] == 128
            measured['ratio_to_rank56'] = matched['throughput_ratio']
            repeats.append(measured)
        cells.append({'rank': rank, 'trainable_parameters': training['trainable_parameters'],
                      'training': training, 'mean_tps': statistics.mean(r['method_tps'] for r in repeats),
                      'timing_stdev_tps': statistics.stdev(r['method_tps'] for r in repeats),
                      'mean_ratio_to_rank56': statistics.mean(r['ratio_to_rank56'] for r in repeats),
                      'per_repeat': repeats, 'source_sha256': hashes})
    assert len(signatures) == 1, 'Runtime or evaluation manifest differs'
    exact = all(r['exact_matches'] == r['finish_matches'] == 128 for c in cells for r in c['per_repeat'])
    return {'status': 'complete', 'all_exact': exact, 'cells': cells,
            'scope': 'One training seed, three timing repetitions; rank56 reused from main study.',
            'cost_note': 'All ranks start from the same three-epoch ZIP fit; add that shared initialization cost.',
            'deployment_note': 'Every adapter is folded into the same dense fusion shape; rank is training capacity.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); result = collect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    assert result['all_exact'], 'Full token/finish mismatch; see report'
