import pytest

from experiments.handoff_transfer.exact_scope.collect import check_transformers_contract


def metadata(mode='auf'):
    return dict(timing_valid=True,contract=dict(
        family='q8',mode=mode,count=128,cap=2048,repeat=0,
        manifest_sha256='manifest',export_sha256=None if mode=='ar' else 'checkpoint',
        runtime_config=dict(backend='transformers',dtype='bfloat16',attention='sdpa',
                            selective_capture=True,temperature=0,target='frozen-target',stop_ids=[1])))


def test_valid_standalone_contract_and_shared_ar():
    ar=check_transformers_contract(metadata('ar'),'ar','manifest',None)
    method=check_transformers_contract(metadata(),'auf','manifest','checkpoint')
    assert ar==method


@pytest.mark.parametrize('key,value',[
    ('export_sha256','another-checkpoint'),('manifest_sha256','another-dataset'),
    ('count',16),('cap',1024),('mode','ce'),('repeat',1),('request_batch_size',128),
])
def test_rejects_wrong_experiment_contract(key,value):
    summary=metadata();summary['contract'][key]=value
    with pytest.raises(ValueError):
        check_transformers_contract(summary,'auf','manifest','checkpoint')


@pytest.mark.parametrize('key,value',[('backend','vllm'),('dtype','float32'),('temperature',.6)])
def test_rejects_wrong_runtime(key,value):
    summary=metadata();summary['contract']['runtime_config'][key]=value
    with pytest.raises(ValueError):
        check_transformers_contract(summary,'auf','manifest','checkpoint')


def test_rejects_invalid_timing():
    summary=metadata();summary['timing_valid']=False
    with pytest.raises(ValueError):
        check_transformers_contract(summary,'auf','manifest','checkpoint')
