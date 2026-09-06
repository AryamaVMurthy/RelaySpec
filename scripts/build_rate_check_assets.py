"""Reconstruct the matched 14B learning-rate comparison from completed evidence."""
import json
import runpy
from pathlib import Path

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest

runpy.run_path('scripts/analyze_research_rate_check.py', run_name='__main__')
fit_path = Path('reports/autoresearch-20260907/rate-check-summary.json')
fits = json.loads(fit_path.read_text())['results']
root = Path('reports/autoresearch-20260907/run-28528')
wave_path = Path('configs/autoresearch/20260907/wave38.json')
wave = json.loads(wave_path.read_text())
ledger = json.loads(Path('reports/autoresearch-20260907/jobs.json').read_text())
job = next(j for j in ledger['jobs'] if j['id'] == 28528)
if (root / 'source-commit.txt').read_text().strip() != job['source_commit']:
    raise ValueError('Unmatched source snapshot')
inputs = {str(p): digest(p) for p in [fit_path, wave_path, root / 'source-commit.txt']}
results = []
lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
         r'Mapper & Records & LR & Original val. & New val. & Throughput ratio \\', r'\midrule']
for i, (spec, fit) in enumerate(zip(wave['lanes'], fits)):
    folder = root / f'lane{i}'
    paths = [root / f'lane{i}-status.json', root / f'lane{i}-spec.json',
             folder / 'mapper-campaign.json', folder / 'completion-gate.json',
             folder / 'benchmark-rank0.jsonl', folder / 'config.yaml']
    status, executed, campaign, gate = [json.loads(p.read_text()) for p in paths[:4]]
    if status['status'] != 'pass' or executed != spec or gate['status'] != 'pass':
        raise ValueError('Incomplete or undeclared lane')
    for cp, sha in spec['checkpoint_sha256'].items():
        found = [v for v in campaign['variants'].values() if v['checkpoint_path'] == cp]
        if len(found) != 1 or found[0]['checkpoint_sha256'] != sha:
            raise ValueError('Frozen mapper hash mismatch')
    rows = [json.loads(x) for x in paths[4].read_text().splitlines()]
    methods = {'relay_base', *spec['candidates']}
    pairs = {(r['problem_id'], r['repetition']) for r in rows}
    if len(pairs) != 8 or len(rows) != 8*len(methods):
        raise ValueError('Incorrect development coverage')
    if {(r['problem_id'],r['repetition'],r['method']) for r in rows} != {(p,rep,m) for p,rep in pairs for m in methods}:
        raise ValueError('Missing or duplicate paired output')
    reference = 'relay_base' if i < 2 else 'relay_original_matched'
    summary = summarize(rows, reference=reference)
    rate = summary['methods']['relay_lower_rate']['throughput_ratio']
    trial = fit['trial']
    architecture = ['Dense','Dense','Linear-4096','MLP-4096'][i]
    lines.append(f"{architecture} & {trial['distinct_examples']:,} & {trial['learning_rate']:g} & "
                 f"{fit['original_reference']['validation_objective']:.3f} & {fit['validation_objective']:.3f} & {rate:.3f}" + r'\\')
    results.append(dict(architecture=architecture, trial=trial, reference=reference, summary=summary,
                        capped_counts={m:sum(r['output_tokens']>=512 for r in rows if r['method']==m) for m in methods}))
    inputs.update({str(p):digest(p) for p in paths})
lines += [r'\bottomrule',r'\end{tabular}']
Path('paper/iclr2027/generated/rate_check_table.tex').write_text('\n'.join(lines)+'\n')
Path('reports/autoresearch-20260907/rate-check-decoding-summary.json').write_text(json.dumps(dict(
    input_sha256=inputs, results=results, scope='Eight exposed MATH questions at a 512-token cap. Each lower-rate endpoint is compared with its original matched architecture/data/seed endpoint. All four checks retained. No independent confirmation or full-answer quality claim.'), indent=2)+'\n')
for result in results:
    print(result['architecture'], result['summary']['methods']['relay_lower_rate']['throughput_ratio'])
