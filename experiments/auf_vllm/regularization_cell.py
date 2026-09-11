"""Weight-decay controls; corresponding zero-decay cells are reused."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from .pilot_data import sha,write
from .scaling_cell import invoke

CELLS=[(kind,loss,wd) for kind in ('dense','rank56') for loss in ('ce','auf') for wd in (.01,.1)]


def main(args):
    kind,loss,wd=CELLS[args.cell];root=args.root
    out=root/'regularization'/f'q8-n512-{kind}-{loss}-wd{wd}-s42'
    extra=[]
    if kind=='rank56':
        base=root/'scaling/records/q8-zip-n512-s42/epoch-3/export/model.safetensors'
        assert base.is_file()
        extra=['--lora-base',base,'--rank',56]
        control=root/'capacity/fusion-residual'/f'n512-r56-{loss}-s42'
    else:control=root/'scaling/records'/f'q8-{loss}-n512-s42'
    assert (control/'summary.json').exists(),control
    out.mkdir(parents=True,exist_ok=True)
    write(out/'design.json',{'records':512,'parameterization':kind,'weight_decay':wd,'objective':loss,
          'zero_decay_control':str(control),'zero_decay_summary_sha256':sha(control/'summary.json'),
          'learning_rate':2e-5 if extra else 1e-3,'epochs':3,
          'initialization_epochs':3 if extra else 0,
          'interpretation':'Weight decay varies within each parameterization; initialization differs between parameterizations'})
    invoke('train_tokens','--data',root/'main-q8-n4096','--validation-data',root/'validation-q8-n1024',
           '--objective',loss,'--records',512,'--epochs',3,'--seed',42,'--lr','2e-5' if extra else '1e-3',
           '--weight-decay',wd,'--defer-validation','--out',out,*extra)
    invoke('evaluate_offline','--fit',out,'--data',root/'validation-q8-n1024')
    case=root/'benchmarks/regularization'/out.name
    case.mkdir(parents=True,exist_ok=True)
    exported=out/'epoch-3/export'
    if (case/'export').exists():assert (case/'export').resolve()==exported.resolve()
    else:(case/'export').symlink_to(exported,target_is_directory=True)
    invoke('prepare_benchmark','--case',case)
    runtime=Path(__file__).parent/'runtime_zip'
    env=dict(os.environ,TRANSFER_WORK=str(case),VLLM_BATCH_INVARIANT='1',OMP_NUM_THREADS='1')
    env['PYTHONPATH']=str(runtime.resolve())+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-u',str(runtime/'benchmark.py'),'mapped','--count','128','--cap','2048',
                    '--tag','development128-cap2048'],env=env,check=True)
    ar=sorted((root/'benchmarks/q8-selected/zip').glob('worker-*/measurements/development128-cap2048/worker_*/ar8-r0.jsonl'))
    assert len(ar)==2
    invoke('compare_outputs','--ar',*ar,'--method',case/'measurements/development128-cap2048/worker_0/mapped-r0.jsonl',
           '--out',case/'comparison.json','--expected-count',128)
    write(out/'cell-complete.json',{'status':'complete','comparison_sha256':sha(case/'comparison.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cell',type=int,choices=range(len(CELLS)),required=True)
    main(p.parse_args())
