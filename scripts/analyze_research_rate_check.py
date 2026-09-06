"""Audit the resumed four-arm EAGLE14 rate check against original fits."""
import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import audit_fit_artifacts, digest

root = Path('reports/autoresearch-20260907/run-28527')
wave_path = Path('configs/autoresearch/20260907/wave37.json')
original_path = Path('reports/mapper-scaling-20260905/target14b-eagle3-fit-results.json')
original = json.loads(original_path.read_text())
wave = json.loads(wave_path.read_text())
inputs = {str(p): digest(p) for p in [wave_path, original_path]}
results = []
for lane, spec in enumerate(wave['lanes']):
    status_path = root / f'lane{lane}-status.json'
    status = json.loads(status_path.read_text())
    if status['status'] != 'pass':
        raise ValueError(f'Lane {lane} has not passed')
    trial = spec['trial']
    folder = root / f'lane{lane}/fitting' / trial['name']
    result = audit_fit_artifacts(folder, trial, cache_sha256=original['feature_cache_index_sha256'])
    original_name = trial['name'].split('-lr')[0]
    reference = original['results'][original_name]
    for key in ['architecture', 'seed', 'steps', 'distinct_examples', 'normalize_input', 'feature_objective']:
        if trial[key] != reference['trial'][key]:
            raise ValueError(f'Unmatched original setting: {key}')
    rows_path = root / f'lane{lane}/benchmark-rank0.jsonl'
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()]
    pairs = {(r['problem_id'], r['repetition']) for r in rows}
    if len(pairs) != 8 or len(rows) != 16 or {r['method'] for r in rows} != {'relay_base', 'relay_reduced'}:
        raise ValueError('Unexpected short-screen coverage')
    if {(r['problem_id'], r['repetition'], r['method']) for r in rows} != {(p, rep, m) for p, rep in pairs for m in ['relay_base','relay_reduced']}:
        raise ValueError('Missing or duplicate method/request pair')
    result['original_reference'] = reference
    result['decoding_screen'] = summarize(rows, reference='relay_base')
    for name, sha in result['source_sha256'].items():
        inputs[str(folder / name)] = sha
    for p in [rows_path, status_path, root / f'lane{lane}-spec.json']:
        inputs[str(p)] = digest(p)
    results.append(result)
output = dict(status='pass', input_sha256=inputs, results=results,
              scope='All four previously declared fixed-endpoint learning-rate checks, one fitting seed. Original fitting references match architecture and records. Eight exposed 128-token requests compare every new arm against original dense512, not a matched-data architecture ranking or full-answer quality confirmation.')
Path('reports/autoresearch-20260907/rate-check-summary.json').write_text(json.dumps(output, indent=2)+'\n')
for r in results:
    print(r['trial']['name'], r['train_objective'], r['validation_objective'],
          r['decoding_screen']['methods']['relay_reduced']['throughput_ratio'])
