"""Run short candidates serially on one GPU, preserving every raw result."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--list', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True,exist_ok=True)
results = []
for config in json.loads(Path(a.list).read_text()):
    run = out/Path(config).stem
    run.mkdir(exist_ok=False)
    env = dict(os.environ,RELAYSPEC_OUTPUT=str(run.resolve()))
    with (run/'stdout.log').open('w') as stdout, (run/'stderr.log').open('w') as stderr:
        subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
            '--nproc_per_node=1','scripts/benchmark_mixed_precision_relay.py',
            '--config',config],env=env,stdout=stdout,stderr=stderr,check=True)
    summary = json.loads((run/'benchmark-summary.json').read_text())
    valid = all(r['exact_sequence_matches'] == r['requests'] > 0
                for r in summary['methods'].values())
    result = {'config':config,'output':str(run),'exact_gate':valid,**summary}
    results.append(result)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(result),flush=True)
(out/'COMPLETE').write_text('All candidates measured; inspect exact_gate before promotion.\n')
