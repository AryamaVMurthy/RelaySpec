import hashlib,json
import pytest
from experiments.handoff_transfer.matrix.collect import collect

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))

def test_collection_rejects_partial_fit_and_changed_runtime(tmp_path):
    run=tmp_path/'cell';base=tmp_path/'baseline';kind='five_ba56'
    write(run/'cell.json',{'kind':kind,'objective':'auf','family':'q8'})
    fitpath=run/'modules/steps-2000'/kind/'summary.json'
    fit={'optimizer_steps':2000,'processed_examples':16000,'anchors_per_example':512,'objective':'auf'}
    write(fitpath,fit)
    model=run/'exports/steps-2000'/kind/'model.safetensors';model.parent.mkdir(parents=True);model.write_bytes(b'test weights')
    sha=hashlib.sha256(model.read_bytes()).hexdigest()
    write(run/'modules/steps-2000'/kind/'verification.json',{'status':'passed','export_sha256':sha})
    normal=base/'normal/export/model.safetensors';normal.parent.mkdir(parents=True);normal.write_bytes(b'normal')
    zipfile=base/'zip/export/model.safetensors';zipfile.parent.mkdir(parents=True);zipfile.write_bytes(b'zip')
    hashes={'normal':hashlib.sha256(b'normal').hexdigest(),'zip':hashlib.sha256(b'zip').hexdigest()}
    write(run/'transfer.json',{'base_export':str(zipfile.parent),'base_sha256':hashes['zip']})
    for rep in range(3):
        for mode,root,wall in [('ar',base,2.),('normal',base,1.5),('zip',base,1.25),('matrix',run,1.)]:
            path=root/'measurements'/f'{mode}-r{rep}-w0.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps({'group_id':str(i),'prompt_ids':[1],'output_ids':[2,3],
                'output_tokens':2,'wall_seconds':wall,'timing_valid':True,'finish_reason':'stop'})+'\n' for i in range(128)))
            contract={'mode':mode,'count':128,'cap':2048,'repeat':rep,'family':'q8','manifest_sha256':'same',
                'export_sha256':sha if mode=='matrix' else hashes.get(mode),'runtime_config':{'dtype':'bfloat16'}}
            write(path.with_suffix('.summary.json'),{'contract':contract,'timing_valid':True,'gpu_before':[{'device_name':'NVIDIA L40S'}]})
    result=collect(run,base)
    assert result['mean_paired_ar_speedup']==2.
    assert result['ratio_to_controls']=={'normal':1.5,'zip':1.25}
    write(fitpath,dict(fit,optimizer_steps=100))
    with pytest.raises(AssertionError):collect(run,base)
    write(fitpath,fit)
    path=run/'measurements/matrix-r1-w0.summary.json';data=json.loads(path.read_text());data['contract']['runtime_config']['dtype']='float16';write(path,data)
    with pytest.raises(AssertionError,match='Unmatched runtime'):collect(run,base)

    data['contract']['runtime_config']['dtype']='bfloat16';write(path,data)
    control=base/'measurements/normal-r0-w0.summary.json';c=json.loads(control.read_text())
    c['contract']['export_sha256']='wrong';write(control,c)
    with pytest.raises(AssertionError,match='Wrong control checkpoint'):collect(run,base)
    c['contract']['export_sha256']=hashes['normal'];c['gpu_before']=[{'device_name':'A100'}];write(control,c)
    with pytest.raises(AssertionError,match='Unmatched GPU hardware'):collect(run,base)
