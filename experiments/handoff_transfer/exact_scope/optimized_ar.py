"""Optimized AR, paired with the prior cohort on the same physical GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.auf_vllm.compare_outputs import compare
from .single_request import ROOT, STUDY, CAMPAIGN

NEW_CAMPAIGN = 'optimized-ar128-20260912'


def run():
    import torch
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    for family in ('q8', 'cross'):
        matches = []
        for worker in range(4):
            old = ROOT / CAMPAIGN / family / f'worker-{worker}'
            summary = json.loads((old / 'ar' / f'ar-r0-w{worker}.summary.json').read_text())
            if summary['gpu_after'][0]['device_uuid'] == uuid:
                matches.append((worker, old, summary))
        assert len(matches) == 1, (family, uuid, matches)
        worker, old, prior = matches[0]
        dest = ROOT / NEW_CAMPAIGN / family / f'worker-{worker}'
        target = prior['contract']['runtime_config']['model']
        command = [sys.executable, '-u', '-m', 'experiments.auf_vllm.family_benchmark',
                   '--family', family, '--models', str(STUDY / 'models'),
                   '--target-path', target, '--data', str(ROOT / 'exact32e1b8-results' / family / 'evaluation'),
                   '--out', str(dest), '--mode', 'ar', '--count', '128', '--cap', '2048',
                   '--workers', '4', '--worker-index', str(worker), '--request-batch-size', '1',
                   '--runtime-profile', 'optimized-ar']
        print(json.dumps({'family': family, 'worker': worker, 'uuid': uuid, 'command': command}), flush=True)
        subprocess.run(command, check=True)
        path = dest / f'ar-r0-w{worker}.jsonl'
        result = compare([old / 'ar' / path.name], [path])
        current = json.loads(path.with_suffix('.summary.json').read_text())
        assert current['gpu_before'][0]['device_uuid'] == current['gpu_after'][0]['device_uuid'] == uuid
        assert current['contract']['manifest_sha256'] == prior['contract']['manifest_sha256']
        assert result['count'] == 32
        record = dict(status='complete', family=family, worker=worker, uuid=uuid,
                      job_id=os.environ['SLURM_JOB_ID'], comparison_to_invariant_ar=result)
        (dest / 'complete.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    run()
