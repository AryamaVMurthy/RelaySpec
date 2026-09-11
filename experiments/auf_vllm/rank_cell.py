"""Matched initialization/data rank ablation for the frozen-base fusion residual."""
import argparse
import json
import os
from pathlib import Path
from .pilot_data import sha, write
from .scaling_cell import invoke
import subprocess
import sys


CELLS=[(n,r,loss) for n in (512,4096) for r in (8,16,32,56,112,224)
       for loss in ('ce','auf') if not (n==4096 and r==32)]


def main(args):
    records,rank,loss=CELLS[args.cell]
    root=args.root
    base=(root/'scaling/records/q8-zip-n512-s42' if records==512 else
          root/'fits/q8-n4096-zip-lr1e-3-s42')/'epoch-3/export/model.safetensors'
    assert base.is_file(),base
    base_contract=json.loads((base.parents[2]/'contract.json').read_text())
    assert base_contract['records']==records and base_contract['epochs']==3
    out=root/'capacity/fusion-residual'/f'n{records}-r{rank}-{loss}-s42'
    out.mkdir(parents=True,exist_ok=True)
    write(out/'design.json',{'records':records,'rank':rank,'loss':loss,'base_sha256':sha(base),
          'unique_training_records_including_initialization':records,
          'initialization_epochs':3,'continuation_epochs':3,'alpha_over_rank':1,
          'trainable_parameters':rank*(20480+2560),'deployed_fusion_parameters':20480*2560,
          'learning_rate':2e-5,'rate_selection':'fixed from rank32 development screen, no per-rank tuning',
          'interpretation':'residual capacity at fixed initialization and passes; merged deployed matrix size unchanged'})
    invoke('train_tokens','--data',root/'main-q8-n4096','--validation-data',root/'validation-q8-n1024',
           '--objective',loss,'--records',records,'--epochs',3,'--lr','2e-5','--seed',42,
           '--lora-base',base,'--rank',rank,'--defer-validation','--out',out)
    invoke('evaluate_offline','--fit',out,'--data',root/'validation-q8-n1024')
    case=root/'benchmarks/capacity/fusion-residual'/out.name
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
    write(out/'cell-complete.json',{'status':'complete','comparison_sha256':sha(case/'comparison.json'),
          'fit_summary_sha256':sha(out/'summary.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cell',type=int,choices=range(len(CELLS)),required=True)
    main(p.parse_args())
