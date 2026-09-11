"""Account for every successful fit in one registered tuning recipe."""
import argparse,json
from pathlib import Path
from .prepare import digest


def collect(runs):
    identity=None;stages={100:[],500:[],2000:[]};sources={}
    for run in runs:
        cell=json.loads((run/'cell.json').read_text())
        key=(cell['family'],cell['kind'],cell['objective'])
        if identity is None:identity=key
        assert key==identity
        for steps in stages:
            path=run/'modules'/f'steps-{steps}'/cell['kind']/'summary.json'
            if not path.exists():continue
            fit=json.loads(path.read_text())
            verification=json.loads(path.with_name('verification.json').read_text())
            assert verification['status']=='passed'
            assert fit['optimizer_steps']==steps and fit['processed_examples']==steps*8
            assert fit['anchors_per_example']==512 and fit['objective']==cell['objective']
            assert fit['lr']==cell['lr'] and fit['training_seconds']>0 and fit['world_size']==2
            stages[steps].append({'lr':cell['lr'],'gpu_hours':fit['training_seconds']*fit['world_size']/3600})
            sources[str(path)]=digest(path)
    assert [len(stages[s]) for s in stages]==[3,2,1], 'Complete 3-to-2-to-1 tuning evidence required'
    assert all(len({r['lr'] for r in rows})==len(rows) for rows in stages.values())
    assert {r['lr'] for r in stages[100]}=={.0001,.0003,.0006}
    assert {r['lr'] for r in stages[2000]}<={r['lr'] for r in stages[500]}<={r['lr'] for r in stages[100]}
    tuning=sum(r['gpu_hours'] for step in (100,500) for r in stages[step])
    final=stages[2000][0]['gpu_hours']
    return dict(identity=identity,successful_fit_gpu_hours=tuning+final,tuning_fit_gpu_hours=tuning,
        final_fit_gpu_hours=final,optimizer_updates_all_successful_fits=3300,
        scope='Successful fit timer totals only; excludes initializer, rollouts, validation, startup/gates, failed attempts and queue waiting',
        stages=stages,sources=sources)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=collect(a.runs);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
