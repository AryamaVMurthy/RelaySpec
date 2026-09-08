"""Freeze selected comparisons and split128 requests into bounded GPU waves."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from confirmation_protocol import file_sha, protocol_signature

root=Path(__file__).resolve().parent
base=json.loads((root/'configs/confirmation-pipeline-smoke.json').read_text())['lanes'][0]
base.update(phase='confirmation',eval_manifest='data/confirmation.json',output_cap=2048,repeats=2,timeout_seconds=540)
provenance=json.loads((root/'reports/run-29162/lane0/provenance.json').read_text())
manifest=json.loads(Path(base['eval_manifest']).read_text())
freeze_path=Path('data/frozen-native-confirmation.json')
if freeze_path.exists():
    raise RuntimeError('Selection already frozen; do not overwrite it')
freeze={
    'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
    'decision':'Selected compact two-tap/five-layer joint checkpoint from29162/lane0, evaluated both as original linear decoding and with DDTree47. Full released-drafter DDTree63 is the stronger external-method reference. Original DFlash is the primary original goal baseline. Selection uses development evidence only. No adaptation is permitted from confirmation results.',
    'selection_evidence':['reports/run-29162/lane0/result.json','reports/run-29363/lane0/compact_two_tap_5layer_ddtree47/result.json','reports/run-29381/lane3/ddtree63_32requests_2048/result.json'],
    'problem_ids':[r['problem_id'] for r in manifest['records']],
    'protocol':protocol_signature(base,root,provenance['target'],provenance['draft'],provenance['source_commit']),
    'interpretation':'No novelty credit for DDTree. Determine whether compact joint training improves original DFlash by at least10%, whether trees add gain, and whether compact+tree improves the full DDTree reference. Intervals cluster questions and exclude fitting-seed uncertainty. AR outputs are collected once for agreement/quality, outside repeated speed measurement.',
}
freeze['selection_evidence_sha256']={p:file_sha(p) for p in freeze['selection_evidence']}
freeze_path.write_text(json.dumps(freeze,indent=2)+'\n')
base.update(frozen_selection=str(freeze_path),frozen_selection_sha256=file_sha(freeze_path))
benchmarks=['gsm8k','math500','humaneval','mtbench']
coverage=[];waves=[]
for wave in range(16):
    lanes=[]
    for lane in range(4):
        benchmark=benchmarks[(lane+wave)%4]
        spec=copy.deepcopy(base)
        spec.update(name=f'confirmation_{wave:02}_{benchmark}',benchmarks=[benchmark],eval_offset=2*wave,requests_per_benchmark=2)
        lanes.append(spec)
        coverage.extend(r['problem_id'] for r in [r for r in manifest['records'] if r['benchmark']==benchmark][2*wave:2*wave+2])
    path=Path(f'configs/confirmation-{wave:02}.json')
    path.write_text(json.dumps({'lanes':lanes},indent=2)+'\n');waves.append(str(path))
assert len(coverage)==128 and len(set(coverage))==128 and set(coverage)==set(freeze['problem_ids'])
Path('data/confirmation-campaign.json').write_text(json.dumps({'status':'frozen_not_run','freeze_sha256':file_sha(freeze_path),'waves':waves,'requests':128,'gpu_limit':4,'timed_methods':['native','compact_linear','candidate','reference'],'timing_repeats':2,'ar_quality_repeats':1},indent=2)+'\n')
print(json.dumps({'freeze_sha256':file_sha(freeze_path),'waves':len(waves),'requests':len(coverage)}))
