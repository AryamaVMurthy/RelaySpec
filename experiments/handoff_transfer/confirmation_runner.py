"""Evaluate frozen primary arms on reserved prompts, with no online selection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from experiments.handoff_transfer.freeze_confirmation import digest
from experiments.auf_vllm.compare_outputs import compare
from experiments.auf_vllm.prepare_workload_benchmark import main as prepare


def validate(protocol, family, workload, repeat):
    assert protocol['status'] == 'frozen' and family in protocol['families']
    evaluation = protocol['evaluation']
    assert workload in evaluation['workloads']
    assert evaluation['requests_per_workload'] == 128 and evaluation['max_output_tokens'] == 2048
    assert evaluation['timing_repetitions'] == 3 and evaluation['natural_eos']
    assert repeat in (0, 1, 2)
    arms = protocol['families'][family]
    expected = {'ar', 'normal', 'zip', 'handoff-r56', 'handoff-five'}
    if family == 'q8':
        expected.add('native')
    assert set(arms) == expected
    for mode, arm in arms.items():
        if mode != 'ar':
            checkpoint = Path(arm['checkpoint'])
            assert digest(checkpoint / 'model.safetensors') == arm['weights_sha256']
            assert digest(checkpoint / 'config.json') == arm['config_sha256']
    return arms


def main(a):
    protocol = json.loads(a.protocol.read_text())
    arms = validate(protocol, a.family, a.workload, a.repeat)
    out = a.out / a.family / a.workload / 'confirmation'
    out.mkdir(parents=True, exist_ok=True)
    link = out / f'protocol-r{a.repeat}.json'
    provenance = {'protocol_path': str(a.protocol), 'protocol_sha256': digest(a.protocol),
                  'family': a.family, 'workload': a.workload, 'repeat': a.repeat}
    if link.exists():
        assert json.loads(link.read_text()) == provenance
    else:
        with link.open('x') as handle:
            handle.write(json.dumps(provenance, indent=2) + '\n')
    # Split materialization occurs only after the protocol and checkpoints pass.
    data = out / f'data-r{a.repeat}'
    prepare(argparse.Namespace(manifest=a.manifest, family=a.family, workload=a.workload,
                               split='confirmation', out=data))
    modes = list(arms)
    modes = modes[a.repeat:] + modes[:a.repeat]
    for mode in modes:
        arm = arms[mode]
        command = [sys.executable, '-u', '-m', 'experiments.auf_vllm.family_benchmark',
            '--family', a.family, '--models', str(a.models), '--target-path', arm['runtime_config']['model'],
            '--data', str(data), '--mode', mode, '--count', '128', '--cap', '2048',
            '--repeat', str(a.repeat), '--out', str(out)]
        if mode != 'ar':
            command += ['--native-draft' if mode == 'native' else '--export', arm['checkpoint']]
        subprocess.run(command, check=True)
        summary = json.loads((out / f'{mode}-r{a.repeat}-w0.summary.json').read_text())
        assert summary['timing_valid']
        assert summary['contract']['runtime_config'] == arm['runtime_config'], 'Runtime changed after freezing'
    evidence = {}
    for mode in modes:
        result = compare([out / f'ar-r{a.repeat}-w0.jsonl'], [out / f'{mode}-r{a.repeat}-w0.jsonl'])
        evidence[mode] = result
    dest = out / f'exactness-r{a.repeat}.json'
    dest.write_text(json.dumps(evidence, indent=2) + '\n')
    assert all(r['count'] == r['exact_matches'] == r['finish_matches'] == 128 for r in evidence.values()), 'Confirmation output mismatch; diagnostics retained'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'manifest', 'models', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--family', choices=('q8', 'q14', 'llama'), required=True)
    p.add_argument('--workload', choices=('math', 'gsm', 'code', 'chat'), required=True)
    p.add_argument('--repeat', type=int, choices=(0, 1, 2), required=True)
    main(p.parse_args())
