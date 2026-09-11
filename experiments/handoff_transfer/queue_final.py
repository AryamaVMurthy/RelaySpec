"""Submit final confirmation and last Transformers replications once, after all checks."""
import json
import subprocess
from pathlib import Path

CONTROL = Path('/home/aryama.murthy/relayspec-auf-20260911')
RECORD = CONTROL / 'outputs/handoff-final-jobs.json'
PREREQUISITES = ['31575', '31576', '31577', '31664', '31665', '31666', '31630',
    '31644', '31657', '31667', '31668', '31673', '31674', '31679', '31684', '31671']


def main():
    assert not RECORD.exists(), 'Final jobs already submitted or partially submitted; inspect recorded IDs before retrying'
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'submitting', 'jobs': [], 'prerequisites': PREREQUISITES}
    def save():
        RECORD.write_text(json.dumps(record, indent=2) + '\n')
    save()
    def submit(stage, family, node, dependencies, script, array=None):
        command = ['sbatch', '--parsable', '--nodelist=' + node,
                   '--dependency=afterok:' + ':'.join(dependencies), '--export=ALL,FAMILY=' + family]
        if array:
            command.append('--array=' + array)
        command.append(str(CONTROL / 'experiments/handoff_transfer' / script))
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        assert job.isdigit(), job
        record['jobs'].append({'stage': stage, 'family': family, 'node': node,
                              'job_id': job, 'dependencies': dependencies, 'array': array})
        save()
        return job
    freezes = {}
    for family, node in (('q8', 'node07'), ('q14', 'node06'), ('llama', 'node06')):
        freezes[family] = submit('freeze', family, node, PREREQUISITES, 'freeze_confirmation.sbatch')
    previous = None
    collectors = []
    for family, node in (('q8', 'node07'), ('q14', 'node06'), ('llama', 'node06')):
        dependencies = list(freezes.values()) if previous is None else [previous, freezes[family]]
        evaluation = submit('confirmation', family, node, dependencies, 'confirmation.sbatch', '0-11%2')
        previous = evaluation
        collectors.append(submit('collect_confirmation', family, node, [evaluation], 'collect_confirmation.sbatch'))
    for family, node in (('q8', 'node07'), ('llama', 'node06')):
        submit('transformers_last', family, node, collectors, 'transformers_final.sbatch')
    record['status'] = 'submitted'
    record['gpu_schedule'] = 'Confirmation families serialized, array maximum2 GPUs. Final two Transformers jobs use one GPU each. All earlier experiment/data lanes must complete first.'
    save()
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
