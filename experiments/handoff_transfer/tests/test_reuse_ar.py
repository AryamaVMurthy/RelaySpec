import hashlib,json
import pytest
from experiments.handoff_transfer.reuse_ar import reuse

def test_reuse_preserves_original_measurement_and_rejects_changed_contract(tmp_path):
    evaluation=tmp_path/'eval.json'
    evaluation.write_text(json.dumps([{'group_id':str(i),'prompt_token_ids':[i+1]} for i in range(128)]))
    contract={'mode':'ar','count':128,'cap':2048,'repeat':0,'manifest_sha256':hashlib.sha256(evaluation.read_bytes()).hexdigest()}
    gate=tmp_path/'gate.summary.json';gate.write_text(json.dumps({'contract':dict(contract,count=4,cap=128)}))
    source=tmp_path/'ar-r0-w0.jsonl'
    source.write_text(''.join(json.dumps({'group_id':str(i),'prompt_ids':[i+1],'timing_valid':True,'wall_seconds':1.,'output_tokens':1,'output_ids':[7],'finish_reason':'stop'})+'\n' for i in range(128)))
    summary=source.with_suffix('.summary.json');summary.write_text(json.dumps({'contract':contract,'timing_valid':True,'job_id':'original'}))
    out=tmp_path/'out';reuse(source,gate,evaluation,out)
    assert (out/source.name).read_bytes()==source.read_bytes()
    assert (out/summary.name).read_bytes()==summary.read_bytes()
    changed=dict(contract,cap=1024);summary.write_text(json.dumps({'contract':changed,'timing_valid':True}))
    with pytest.raises(AssertionError,match='contract differs'):reuse(source,gate,evaluation,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
