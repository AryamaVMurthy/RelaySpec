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

def test_zip_reuse_requires_matching_export_bytes(tmp_path):
    evaluation=tmp_path/'eval.json'
    evaluation.write_text(json.dumps([{'group_id':str(i),'prompt_token_ids':[i+1]} for i in range(128)]))
    export=tmp_path/'export';export.mkdir()
    (export/'config.json').write_text(json.dumps({'block_size':16}))
    weights=export/'model.safetensors';weights.write_bytes(b'checkpoint-for-contract-test')
    ar={'mode':'ar','count':4,'cap':128,'repeat':0,'manifest_sha256':hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        'runtime_config':{'dtype':'bfloat16','speculative_config':None}}
    gate=tmp_path/'gate.summary.json';gate.write_text(json.dumps({'contract':ar}))
    contract=dict(ar,mode='zip',count=128,cap=2048,export_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        runtime_config={'dtype':'bfloat16','speculative_config':{'method':'dflash','model':str(export),'num_speculative_tokens':15}})
    source=tmp_path/'zip-r0-w0.jsonl'
    source.write_text(''.join(json.dumps({'group_id':str(i),'prompt_ids':[i+1],'timing_valid':True,'wall_seconds':1.,'output_tokens':1,'output_ids':[7],'finish_reason':'stop'})+'\n' for i in range(128)))
    source.with_suffix('.summary.json').write_text(json.dumps({'contract':contract,'timing_valid':True}))
    reuse(source,gate,evaluation,tmp_path/'good',mode='zip',base_export=export)
    weights.write_bytes(b'different checkpoint')
    with pytest.raises(AssertionError,match='contract differs'):
        reuse(source,gate,evaluation,tmp_path/'bad',mode='zip',base_export=export)
