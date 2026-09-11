"""Matched Qwen five-BA serving study, after latency finalists are frozen."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from experiments.auf_vllm.compare_outputs import compare
from .prepare import digest


def frozen_exports(root):
    base=root/'q8';matrix=root/'matrix/q8'
    exports={'ar':None,'normal':base/'normal/export'}
    transfer=json.loads((base/'transfer.json').read_text());exports['zip']=Path(transfer['base_export'])
    for loss in ('ce','auf'):
        candidates=list(matrix.glob(f'five_ba56-{loss}-lr*/final-comparison.json'))
        assert len(candidates)==1, 'Need exactly one finalized validation-selected LR per loss'
        result=json.loads(candidates[0].read_text());assert result['status']=='complete_verified'
        assert result['cell']['family']=='q8' and result['cell']['kind']=='five_ba56' and result['cell']['objective']==loss
        for control in ('normal','zip'):
            assert digest(exports[control]/'model.safetensors')==result['control_export_sha256'][control], 'Control differs from main evaluation'
        export=candidates[0].parent/'exports/steps-2000/five_ba56'
        assert digest(export/'model.safetensors')==result['verification']['export_sha256']
        exports[loss]=export
    return exports


def main(a):
    base=a.root/'q8';exports=frozen_exports(a.root)
    a.out.mkdir(parents=True,exist_ok=True)
    evidence=[]
    for batch in (4,8):
        for repeat in range(3):
            paths={}
            for mode,export in exports.items():
                cmd=[sys.executable,'-m','experiments.auf_vllm.family_benchmark','--family','q8',
                    '--target-path',str(a.target),'--models',str(a.models),'--data',str(base/'evaluation'),
                    '--mode',mode,'--count','128','--cap','2048','--repeat',str(repeat),
                    '--request-batch-size',str(batch),'--out',str(a.out)]
                if export is not None:cmd+=['--export',str(export)]
                subprocess.run(cmd,check=True)
                paths[mode]=a.out/f'{mode}-r{repeat}-w0-b{batch}.jsonl'
            comparisons={}
            for mode in ('normal','zip','ce','auf'):
                result=compare([paths['ar']],[paths[mode]])
                assert result['count']==result['exact_matches']==result['finish_matches']==128
                comparisons[mode]=result
            evidence.append(dict(batch=batch,repeat=repeat,comparisons=comparisons,
                sources={str(item):digest(item) for p in paths.values() for item in (p,p.with_suffix('.summary.json'))}))
            (a.out/f'comparison-b{batch}-r{repeat}.json').write_text(json.dumps(evidence[-1],indent=2)+'\n')
    (a.out/'complete.json').write_text(json.dumps(dict(scope='Qwen fiveBA CE/AUF fixed-batch serving',
        status='complete',comparisons=evidence),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','models','target','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
