"""Collect all reserved workloads without reclassifying development results."""
import argparse
import json
from pathlib import Path
from experiments.handoff_transfer.freeze_confirmation import digest
from experiments.handoff_transfer.collect_transfer import main as collect


def main(a):
    protocol = json.loads(a.protocol.read_text())
    assert protocol['status'] == 'frozen'
    modes = list(protocol['families'][a.family])
    for workload in protocol['evaluation']['workloads']:
        measured = a.root / a.family / workload / 'confirmation'
        for repeat in range(3):
            link = json.loads((measured / f'protocol-r{repeat}.json').read_text())
            assert link['protocol_sha256'] == digest(a.protocol)
            assert link['family'] == a.family and link['workload'] == workload and link['repeat'] == repeat
            for mode in modes:
                summary = json.loads((measured / f'{mode}-r{repeat}-w0.summary.json').read_text())
                assert summary['contract']['runtime_config'] == protocol['families'][a.family][mode]['runtime_config']
        output = a.out / f'{a.family}-{workload}-confirmation.json'
        collect(argparse.Namespace(root=measured, family=a.family, measurements=measured,
            out=output, methods=modes, evaluation=f'128 reserved {workload} confirmation requests, cap2048, natural EOS, greedy'))
        report = json.loads(output.read_text())
        report['protocol_sha256'] = digest(a.protocol)
        report['uncertainty'] = 'Three timing repetitions of frozen seed42 primary checkpoints on reserved prompts; fitting-seed results reported separately.'
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--family', choices=('q8', 'q14', 'llama'), required=True)
    main(p.parse_args())
