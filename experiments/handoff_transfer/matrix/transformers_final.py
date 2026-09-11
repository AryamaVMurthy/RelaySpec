"""Standalone Transformers confirmation of the finalized Qwen five-BA pair."""
import argparse,json,subprocess,sys
from pathlib import Path
from .batch_final import frozen_exports
from .prepare import digest
from experiments.auf_vllm.compare_outputs import compare

def main(a):
    exports=frozen_exports(a.root)
    for phase,count,cap in [('gate',4,128),('full',128,2048)]:
        out=a.out/phase;out.mkdir(parents=True,exist_ok=True);results={}
        for mode,export in exports.items():
            command=[sys.executable,'-m','experiments.handoff_transfer.transformers_benchmark',
                '--family','q8','--target',str(a.target),'--data',str(a.root/'q8/evaluation'),
                '--out',str(out),'--dflash-source',str(a.dflash_source),'--mode',mode,'--count',str(count),'--cap',str(cap)]
            if export is not None:command+=['--export',str(export)]
            subprocess.run(command,check=True)
            path=out/f'{mode}-r0-w0.jsonl'
            summary=json.loads(path.with_suffix('.summary.json').read_text())
            assert summary['contract']['export_sha256']==(digest(export/'model.safetensors') if export else None)
            if mode!='ar':
                result=compare([out/'ar-r0-w0.jsonl'],[path])
                assert result['count']==result['exact_matches']==result['finish_matches']==count
                results[mode]=result
        (out/'comparisons.json').write_text(json.dumps(dict(scope='Standalone Transformers confirmation; one timing per checkpoint',results=results),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','target','dflash-source','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
