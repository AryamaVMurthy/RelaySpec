"""Eight-request Qwen native/AR diagnostic, four fixed numerical profiles."""
import json
import os
from pathlib import Path
import subprocess
import sys
from .single_request import ROOT
from experiments.auf_vllm.compare_outputs import compare

CAMPAIGN='exactness-search-20260912'
PROFILES=('stock','invariant-o3','strict-blas','invariant-smalltile',
          'invariant-o3-rms','invariant-o3-all','invariant-smalltile-rms','invariant-smalltile-all')


def run():
    profile=os.environ['RELAYSPEC_NUMERICS'];assert profile in PROFILES
    root=ROOT/CAMPAIGN/profile;root.mkdir(parents=True,exist_ok=True)
    if profile.startswith('invariant-smalltile'):
        from experiments.auf_vllm.numerics_runtime import kernel_check
        (root/'kernel-check.json').write_text(json.dumps(kernel_check(),indent=2)+'\n')
    data=root/'data';data.mkdir(exist_ok=True)
    source=ROOT/'exact32e1b8-results/q8/evaluation'
    for name,number in [('eval.json',8),('warmup.json',1)]:
        (data/name).write_text(json.dumps(json.loads((source/name).read_text())[:number]))
    target=ROOT.parent/'transfer-reproduction-20260907/work/models/8b/target'
    draft=target.parent/'draft'
    paths={};effective=None
    for mode in ['ar','native']:
        extra=[] if mode=='ar' else ['--native-draft',draft]
        cmd=[sys.executable,'-u','-m','experiments.auf_vllm.family_benchmark','--family','q8',
             '--models',target.parent,'--target-path',target,'--data',data,'--out',root/mode,
             '--mode',mode,'--count',8,'--cap',512,'--workers',1,'--worker-index',0,
             '--request-batch-size',1,'--runtime-profile','numerics',*extra]
        subprocess.run(list(map(str,cmd)),check=True)
        path=root/mode/f'{mode}-r0-w0.jsonl';paths[mode]=path
        s=json.loads(path.with_suffix('.summary.json').read_text())
        if effective is None:effective=s['effective_runtime']
        assert effective==s['effective_runtime'],(effective,s['effective_runtime'])
    result=compare([paths['ar']],[paths['native']])
    result.update(profile=profile,effective_runtime=effective,job_id=os.environ['SLURM_JOB_ID'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':run()
