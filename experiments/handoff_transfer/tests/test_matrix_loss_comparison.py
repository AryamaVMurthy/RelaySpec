from copy import deepcopy
import pytest
from experiments.handoff_transfer.matrix.loss_comparison import paired


def result(loss,tps):
    return dict(status='complete_verified',cell=dict(family='q8',kind='five_maps',objective=loss,lr=.0001),
        fit=dict(optimizer_steps=2000,processed_examples=16000,anchors_per_example=512),
        gpu_hardware=['L40S'],control_export_sha256={'normal':'n','zip':'z'},
        sources={f'/baseline/ar-r{i}-w0{suffix}':'abc' for i in range(3) for suffix in ('.jsonl','.summary.json')},mean_tps=tps,
        repeats=[dict(count=128,exact_matches=128,finish_matches=128,method_output_tokens=10000,method_tps=tps)]*3)


def test_paired_loss_and_evidence_checks():
    ce,auf=result('ce',100),result('auf',110)
    assert paired([ce,auf])[0]['mean_auf_over_ce']==1.1
    with pytest.raises(AssertionError,match='Both final'):paired([ce])
    other=deepcopy(auf);other['sources']['/baseline/ar-r0-w0.jsonl']='changed'
    with pytest.raises(AssertionError,match='Different AR'):paired([ce,other])
    other=deepcopy(auf);other['repeats'][0]['exact_matches']=127
    with pytest.raises(AssertionError):paired([ce,other])
