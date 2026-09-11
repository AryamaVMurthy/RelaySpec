"""Verify and summarize full package reproduction against immutable references."""
import argparse,collections,hashlib,json,random,statistics
from pathlib import Path

def main(a):
    fit_path=a.root.parent/'origin-full-fit/summary.json'
    fit=json.loads(fit_path.read_text())
    gate_path=a.root.parent/'origin-full-fit/gate-check.json'
    gate=json.loads(gate_path.read_text())
    assert fit['optimizer_steps']==2000 and fit['processed_examples']==16000
    assert gate['status']=='passed' and gate['all_non_fusion_weights_exact'] and gate['unique_cached_records']==4096
    refs={r['group_id']:r for r in json.loads(a.references.read_text())}
    methods={};hashes={}
    for mode,file in [('ar','ar'),('native','native'),('mapper','mapped')]:
        p=a.root/mode/f'{file}.json';data=json.loads(p.read_text())
        assert data['complete'] and data['cap']==2048 and data['batch_invariant']
        assert data['backend']=='vllm' and data['compilation_mode']==0 and data['cudagraph_mode']=='FULL_DECODE_ONLY'
        rows={r['group_id']:r for r in data['rows']}
        assert len(rows)==len(data['rows'])==128
        for k,r in rows.items():
            assert (r['output_ids'],r['finish_reason'])==(refs[k]['output_ids'],refs[k]['finish_reason'])
            assert r['seconds']>0
        methods[mode]=rows;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert methods['ar'].keys()==methods['native'].keys()==methods['mapper'].keys()
    rows={}
    for mode,values in methods.items():
        tokens=sum(len(r['output_ids']) for r in values.values());seconds=sum(r['seconds'] for r in values.values())
        rows[mode]={'tps':tokens/seconds,'output_tokens':tokens,'request_wall_seconds':seconds,
            'exact_token_matches':128,'exact_finish_matches':128,
            'finish_reasons':dict(collections.Counter(r['finish_reason'] for r in values.values()))}
        if mode!='ar':
            iterations=sum(r['verification_iterations'] for r in values.values())
            rows[mode]['accepted_draft_tokens_per_verification']=sum(r['accepted_draft_tokens'] for r in values.values())/iterations
    rng=random.Random(42);keys=list(methods['ar']);ratios=[]
    for _ in range(2000):
        sample=rng.choices(keys,k=128)
        ratios.append(sum(methods['native'][k]['seconds'] for k in sample)/sum(methods['mapper'][k]['seconds'] for k in sample))
    ratios.sort()
    for p in [fit_path,gate_path]:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    result={'status':'verified_complete','setting':'Original handoff Math, Qwen3-4B with pinned frozen target LoRA; native Qwen3-4B DFlash; train fusion LoRA only',
        'requests':128,'max_output_tokens':2048,'training_records':4096,'optimizer_steps':2000,
        'trainable_parameters':860160,'training_seconds':fit['training_seconds'],
        'training_seeds':[42],'timing_repetitions':1,'rows':rows,
        'mapper_vs_native_tps_ratio':rows['mapper']['tps']/rows['native']['tps'],
        'mapper_vs_ar_tps_ratio':rows['mapper']['tps']/rows['ar']['tps'],
        'paired_request_bootstrap_95_ratio_interval':[ratios[49],ratios[1949]],
        'interval_limit':'Request variation only; excludes fitting-seed and timing-repetition variation.',
        'scope_limit':'Original package reproduction only; does not establish frozen-original-target transfer gains.',
        'source_sha256':hashes,'references_sha256':hashlib.sha256(a.references.read_bytes()).hexdigest()}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
