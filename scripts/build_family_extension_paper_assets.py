"""Rebuild the family and rollout extension tables from verified raw requests."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
import yaml


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ci_ratio(a, b, seed=1729):
    """Paired time ratio, valid because compared arms have equal token arrays."""
    a, b = np.array(a), np.array(b)
    assert len(a) == len(b) == 128
    ids = np.random.default_rng(seed).integers(0, 128, (10000, 128))
    return [float(x) for x in np.quantile(a[ids].sum(1)/b[ids].sum(1), [.025, .975])]


def build(root, output):
    inputs = {}
    def read(p):
        p = root / p
        inputs[str(p.relative_to(root))] = digest(p)
        return p.read_text()
    base = Path('reports/family-eval128-20260908')
    manifest = json.loads(read(base/'manifest.json'))['records']
    expected = {r['problem_id'] for r in manifest}
    assert len(expected) == len(manifest) == 128
    groups = {(f,m): {} for f in ['llama','cross'] for m in ['ar','old','selected']}
    for wave in range(16):
        for lane in range(4):
            folder = base/'artifacts'/f'wave{wave}'/f'lane{lane}'
            read(folder/'COMPLETE')
            cfg = yaml.safe_load(read(folder/'shared/config.yaml'))
            assert cfg['generation']['max_new_tokens'] == 2048
            assert cfg['benchmark']['precision'] == cfg['benchmark']['target_head_precision'] == 'float32'
            assert cfg['benchmark']['block_size'] == (10 if lane < 2 else 16)
            assert cfg['target']['id'] == ('unsloth/Llama-3.2-3B-Instruct' if lane < 2 else 'unsloth/Llama-3.1-8B-Instruct')
            assert cfg['source_trunk']['model']['id'] == ('unsloth/Llama-3.1-8B-Instruct' if lane < 2 else 'Qwen/Qwen3-4B')
            assert cfg['generation']['temperature'] == 0.0
            assert cfg['benchmark']['unload_source_trunk'] is True
            precision = json.loads(read(folder/'shared/mixed-precision.json'))
            assert precision['target'] == precision['mapper'] == 'float32'
            assert precision['drafter'] == 'bfloat16' and precision['verifier_changed'] is False
            rows = [json.loads(s) for s in read(folder/'shared/benchmark-rank0.jsonl').splitlines()]
            assert len(rows) == 12
            for r in rows:
                name = {'native_ar':'ar','relay_p':'old','relay_p_cross_family':'old','relay_candidate_1':'selected'}[r['method']]
                g = groups['llama' if lane < 2 else 'cross', name]
                k = r['problem_id']
                assert k in expected and k not in g and r['repetition'] == r['turn_index'] == 0
                ids = r['output_token_ids']
                assert len(ids) == r['output_tokens'] <= 2048
                assert hashlib.sha256(struct.pack('<'+'i'*len(ids), *ids)).hexdigest() == r['output_hash']
                assert r['request_seconds'] >= r['decode_seconds'] > 0
                g[k] = r
    family = {}
    table = [r'\begin{tabular}{llrrrr}', r'\toprule', r'Drafter source $\rightarrow$ target & Fit records & AR tok/s & Relay tok/s & $T_R/T_A$ & Exact AR \\', r'\midrule']
    for f, label in [('llama',r'Llama-8B $\rightarrow$ Llama-3B'), ('cross',r'Qwen-4B $\rightarrow$ Llama-8B')]:
        ar = groups[f,'ar']; assert set(ar) == expected
        keys = sorted(expected); family[f] = {}
        for m in ['ar','old','selected']:
            rows = groups[f,m]; assert set(rows) == expected
            matches = sum(rows[k]['output_token_ids'] == ar[k]['output_token_ids'] for k in keys)
            assert matches == 128
            n = sum(r['output_tokens'] for r in rows.values())
            sec = sum(r['request_seconds'] for r in rows.values())
            dec = sum(r['decode_seconds'] for r in rows.values())
            family[f][m] = {'tokens':n, 'request_tps':n/sec, 'decode_tps':n/dec, 'seconds':sec, 'exact_matches':matches, 'cap_reached':sum(r['output_tokens']==2048 for r in rows.values())}
        a,o,s = [family[f][m] for m in ['ar','old','selected']]
        family[f]['selected_over_old'] = {'ratio':o['seconds']/s['seconds'], 'ci':ci_ratio([groups[f,'old'][k]['request_seconds'] for k in keys], [groups[f,'selected'][k]['request_seconds'] for k in keys])}
        for m, display in [('old','4,096'),('selected','16,384' if f == 'llama' else '8,192')]:
            r=family[f][m]
            r['ar_ratio_ci'] = ci_ratio([ar[k]['request_seconds'] for k in keys], [groups[f,m][k]['request_seconds'] for k in keys])
            table.append(f"{label} & {display} & {a['request_tps']:.2f} & {r['request_tps']:.2f} & {r['request_tps']/a['request_tps']:.2f} & 128/128 " + r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}'])
    gen = output/'generated'; gen.mkdir(parents=True,exist_ok=True)
    (gen/'family128_table.tex').write_text('\n'.join(table)+'\n')
    transfer=[]
    for repeat in range(2):
        base2=Path('reports/transfer-reproduction-20260907/full16384-results')
        recorded=json.loads(read(base2/f'final-comparison-r{repeat}.json'))
        source_audit=json.loads(read(base2/'source/source-audit.json'))
        assert source_audit['all_other_python_sources_identical'] is True
        for source_path, expected_digest in source_audit['sha256'].items():
            read(base2/'source'/source_path)
            assert inputs[str(base2/'source'/source_path)] == expected_digest
        methods={}
        for name, paths in recorded['sources'].items():
            rows={}
            for entry in paths:
                path=Path(entry['path']); text=read(path)
                assert inputs[str(path)] == entry['sha256']
                meta_path=path.with_suffix('.summary.json')
                meta=json.loads(read(meta_path))
                assert meta['args']['count'] == 128 and meta['args']['cap'] == 2048
                assert meta['args']['split'] == 'eval' and meta['mode'] == name
                assert meta['config']['dtype'] == 'bfloat16'
                assert meta['config']['enable_prefix_caching'] is False
                assert meta['timing_valid'] is True
                assert str(meta['cuda_visible_devices']) == entry['gpu']
                if name != 'ar8':
                    assert meta['config']['speculative_config']['num_speculative_tokens'] == 15
                for line in text.splitlines():
                    r=json.loads(line); assert r['group_id'] not in rows
                    assert r['timing_valid'] and len(r['output_ids']) == r['output_tokens']
                    rows[r['group_id']]=r
            assert len(rows)==128
            methods[name]=rows
        keys=sorted(methods['ar8'])
        for rows in methods.values():
            assert set(rows)==set(keys)
            for k in keys:
                for field in ['prompt_ids','output_ids','finish_reason']:
                    assert rows[k][field]==methods['ar8'][k][field]
        stats={m:{'tokens':sum(r['output_tokens'] for r in rr.values()),'seconds':sum(r['wall_seconds'] for r in rr.values())} for m,rr in methods.items()}
        for m,s in stats.items():
            s['tps']=s['tokens']/s['seconds']
            assert abs(s['tps']-recorded['stats'][m]['tps'])<1e-9
        transfer.append({'repeat':repeat,'stats':stats,'mapped_over_native':stats['native8']['seconds']/stats['mapped']['seconds'],'ci':ci_ratio([methods['native8'][k]['wall_seconds'] for k in keys],[methods['mapped'][k]['wall_seconds'] for k in keys],42),'exact_matches':128})
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Run & AR tok/s & Native tok/s & Relay tok/s & Relay/native [95\% CI] \\',r'\midrule']
    for r in transfer:
        stats=r['stats']; low,high=r['ci']
        label='Original assignment' if r['repeat']==0 else 'Reversed GPUs'
        lines.append(f"{label} & {stats['ar8']['tps']:.2f} & {stats['native8']['tps']:.2f} & {stats['mapped']['tps']:.2f} & {r['mapped_over_native']:.3f} [{low:.3f}, {high:.3f}] "+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (gen/'rollout_transfer_table.tex').write_text('\n'.join(lines)+'\n')
    summary={'family128':family,'rollout_transfer':transfer,'inputs':inputs,'bootstrap':'10000 paired request resamples; conditional on checkpoint/runtime, no fitting-seed or repeated-run uncertainty'}
    (gen/'family_extension_evidence.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,default=Path('paper/iclr2027'));a=p.parse_args()
    s=build(a.root,a.output)
    print(json.dumps({k:v for k,v in s.items() if k!='inputs'},indent=2))
