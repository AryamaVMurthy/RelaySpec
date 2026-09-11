"""Select within one architecture/loss using matched tuning measurements only."""
import argparse,json
from pathlib import Path
from experiments.auf_vllm.compare_outputs import compare
from experiments.handoff_transfer.matrix.prepare import digest

def select(runs,steps):
    candidates=[];identity=None;manifest=None;runtime=None
    for run in runs:
        cell=json.loads((run/'cell.json').read_text())
        current=(cell['family'],cell['kind'],cell['objective'])
        if identity is None:identity=current
        assert current==identity,'Cannot tune across different architecture/loss cells'
        folder=run/f'screen-{steps}'
        path=folder/'matrix-r0-w0.jsonl';ar=folder/'ar-r0-w0.jsonl'
        contract=json.loads(path.with_suffix('.summary.json').read_text())['contract']
        assert contract['count']==32 and contract['cap']==512 and contract['repeat']==0
        if manifest is None:manifest=contract['manifest_sha256']
        assert contract['manifest_sha256']==manifest,'Different validation prompts'
        config=dict(contract['runtime_config']);config.pop('speculative_config',None)
        if runtime is None:runtime=config
        assert config==runtime,'Different tuning runtime'
        ar_contract=json.loads(ar.with_suffix('.summary.json').read_text())['contract']
        ar_config=dict(ar_contract['runtime_config']);ar_config.pop('speculative_config',None)
        assert ar_config==config and ar_contract['manifest_sha256']==manifest
        assert ar_contract['count']==32 and ar_contract['cap']==512
        fit=run/'modules'/f'steps-{steps}'/cell['kind']
        summary=json.loads((fit/'summary.json').read_text())
        verification=json.loads((fit/'verification.json').read_text())
        assert summary['optimizer_steps']==steps and summary['anchors_per_example']==512
        assert summary['objective']==cell['objective'] and summary['lr']==cell['lr']
        export=run/'exports'/f'steps-{steps}'/cell['kind']/'model.safetensors'
        assert verification['status']=='passed'
        assert verification['export_sha256']==contract['export_sha256']==digest(export)
        evidence=compare([ar],[path])
        assert evidence['exact_matches']==evidence['finish_matches']==32
        candidates.append({'run':str(run),'lr':cell['lr'],'tps':evidence['method_tps'],
            'ar_ratio':evidence['throughput_ratio'],'checkpoint_sha256':digest(export),
            'rows_sha256':digest(path),'training_seconds':summary['training_seconds']})
    assert candidates and len({x['lr'] for x in candidates})==len(candidates)
    ranked=sorted(candidates,key=lambda x:(-x['ar_ratio'],x['lr']))
    return {'scope':'tuning only; not final evaluation','identity':identity,'steps':steps,
        'manifest_sha256':manifest,'ranking_rule':'descending paired aggregate TPS/AR, lower LR for exact ties',
        'ranked':ranked,'selected_lr':ranked[0]['lr'],'limitations':'best observed candidate at this budget; single timing measurement'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=select(a.runs,a.steps);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
