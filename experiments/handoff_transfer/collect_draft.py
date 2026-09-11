"""Compare the matched body-LoRA control with all primary interface baselines."""
import argparse
import json
from pathlib import Path
from experiments.handoff_transfer.collect_transfer import main

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    body = a.root / 'q8-draft-control-r32'
    training = json.loads((body / 'modules/full-draft_lora/draft_lora/summary.json').read_text())
    verification = json.loads((body / 'modules/full-draft_lora/draft_lora/verification.json').read_text())
    assert training['optimizer_steps'] == 2000 and training['processed_examples'] == 16000
    assert verification['status'] == 'passed' and verification['frozen_interface_exact']
    main(argparse.Namespace(root=a.root / 'q8', out=a.out, family='q8',
        methods=['ar', 'normal', 'zip', 'handoff-r56', 'handoff-five', 'handoff-draft'],
        method_roots={'handoff-draft': body / 'measurements'}))
    report = json.loads(a.out.read_text())
    report.update(drafter_training=training, drafter_verification=verification,
        comparison_scope='Same 4096-record ZIP initialization and handoff AUF schedule; rank32 body LoRA is not parameter-matched to fusion rank56.')
    a.out.write_text(json.dumps(report, indent=2) + '\n')
