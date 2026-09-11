"""Execute fixed-update or continuous-epoch fits and complete endpoint evaluations."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .pilot_data import write,sha


def invoke(module,*values):
    subprocess.run([sys.executable,'-u','-m','experiments.auf_vllm.'+module,*map(str,values)],check=True)


def main(args):
    root=args.root;data=root/'main-q8-n4096';valid=root/'validation-q8-n1024'
    out=root/'scaling'/args.axis/f'q8-{args.loss}-n{args.records}-s42'
    if args.axis=='updates':
        while True:
            summary=out/'summary.json'
            if summary.exists() and json.loads(summary.read_text())['step']==1024:break
            invoke('fit_updates','--objective',args.loss,'--data',data,'--validation-data',valid,
                   '--out',out,'--records',args.records,'--updates',1024,'--chunk-updates',128,'--lr','1e-3')
        endpoints=[1024];label='step'
    else:
        invoke('train_tokens','--objective',args.loss,'--data',data,'--validation-data',valid,
               '--out',out,'--records',args.records,'--epochs',12,'--lr','1e-3','--defer-validation')
        endpoints=[1,3,6,12];label='epoch'
    invoke('evaluate_offline','--fit',out,'--data',valid,'--'+label+'s',*endpoints)
    for endpoint in endpoints:
        checkpoint=f'{label}-{endpoint}'
        case=root/'benchmarks/scaling'/args.axis/f'q8-{args.loss}-n{args.records}'/checkpoint
        case.mkdir(parents=True,exist_ok=True)
        exported=out/checkpoint/'export';link=case/'export'
        if link.exists():assert link.resolve()==exported.resolve()
        else:link.symlink_to(exported,target_is_directory=True)
        invoke('prepare_benchmark','--case',case)
        env=dict(os.environ,TRANSFER_WORK=str(case),VLLM_BATCH_INVARIANT='1',OMP_NUM_THREADS='1')
        runtime=Path(__file__).parent/'runtime_zip'
        env['PYTHONPATH']=str(runtime.resolve())+os.pathsep+env.get('PYTHONPATH','')
        subprocess.run([sys.executable,'-u',str(runtime/'benchmark.py'),'mapped','--count','128','--cap','2048',
                        '--tag','development128-cap2048'],env=env,check=True)
        ar=sorted((root/'benchmarks/q8-selected/zip').glob('worker-*/measurements/development128-cap2048/worker_*/ar8-r0.jsonl'))
        assert len(ar)==2
        invoke('compare_outputs','--ar',*ar,'--method',case/'measurements/development128-cap2048/worker_0/mapped-r0.jsonl',
               '--out',case/'comparison.json','--expected-count',128)
    write(out/'cell-complete.json',{'axis':args.axis,'records':args.records,'loss':args.loss,
          'endpoints':endpoints,'fit_summary_sha256':sha(out/'summary.json'),
          'evaluation':'128 Numina development requests, cap2048, one timing repetition per endpoint',
          'ar_reference':'Same previously recorded128 requests; independent full main timing repetitions are separate',
          'status':'complete'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--axis',choices=['updates','epochs'],required=True)
    p.add_argument('--loss',choices=['ce','auf'],required=True)
    p.add_argument('--records',type=int,required=True)
    main(p.parse_args())
