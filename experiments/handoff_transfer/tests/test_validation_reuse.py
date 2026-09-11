import json
import pytest
from experiments.handoff_transfer.matrix.reuse_validation import reuse
from experiments.handoff_transfer.matrix.prepare import digest


def test_reuse_preserves_measured_rows_and_rejects_changed_prompts(tmp_path):
    data=tmp_path/'data';data.mkdir();source=tmp_path/'ar.jsonl'
    (data/'eval.json').write_text(json.dumps([dict(group_id=str(i),prompt_token_ids=[i]) for i in range(32)]))
    source.write_text(''.join(json.dumps(dict(group_id=str(i),prompt_ids=[i],output_ids=[1],output_tokens=1,
        wall_seconds=.2,timing_valid=True,finish_reason='stop'))+'\n' for i in range(32)))
    contract=dict(mode='ar',family='q8',count=32,cap=512,repeat=0,workers=1,worker_index=0,
        export_sha256=None,runtime_config={'model':'/model'},manifest_sha256=digest(data/'eval.json'))
    source.with_suffix('.summary.json').write_text(json.dumps(dict(contract=contract,timing_valid=True,job_id='original')))
    out=tmp_path/'out';result=reuse(source,data,out,'q8',tmp_path.__class__('/model'))
    assert result['counts_as_new_timing_repetitions']==0
    assert (out/'ar-r0-w0.jsonl').read_bytes()==source.read_bytes()
    assert json.loads((out/'ar-r0-w0.summary.json').read_text())['job_id']=='original'
    assert reuse(source,data,out,'q8',tmp_path.__class__('/model'))['status']=='retained_existing_measurement'
    (data/'eval.json').write_text('[]')
    with pytest.raises(AssertionError):reuse(source,data,tmp_path/'bad','q8',tmp_path.__class__('/model'))
