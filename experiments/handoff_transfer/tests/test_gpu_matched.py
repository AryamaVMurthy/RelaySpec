import copy
import json
import pytest
from experiments.handoff_transfer.exact_scope.matched import require_same_device
from experiments.handoff_transfer.exact_scope.gpu_references import required_devices


def summary():
    return dict(gpu_before=[dict(device_uuid='card-a')],gpu_after=[dict(device_uuid='card-a')],timing_valid=True,
                contract=dict(family='q8',manifest_sha256='prompts',count=128,cap=2048,repeat=0,
                              request_batch_size=128,runtime_config=dict(model='target')))


def test_identical_export_does_not_make_different_gpu_timings_matched():
    method=summary();reference=copy.deepcopy(method)
    assert require_same_device(method,reference)=='card-a'
    reference['gpu_after'][0]['device_uuid']='card-b'
    with pytest.raises(ValueError,match='different physical GPUs'):require_same_device(method,reference)


def test_same_gpu_still_requires_same_workload_and_repeat():
    method=summary();reference=summary();reference['contract']['repeat']=1
    with pytest.raises(ValueError,match='repeat'):require_same_device(method,reference)


def test_reference_plan_uses_actual_cards_across_all_repetitions(tmp_path):
    (tmp_path/'evaluation-batch.json').write_text(json.dumps(dict(request_batch_size=128)))
    for index in range(7):
        folder=tmp_path/f'cell-{index}';folder.mkdir()
        (folder/'complete.json').write_text(json.dumps(dict(status='complete',timing_repetitions=3)))
        for repeat in range(3):
            record=summary()
            if index==6 and repeat==2:
                for key in ('gpu_before','gpu_after'):record[key][0]['device_uuid']='card-b'
            (folder/f'matrix-r{repeat}-w0-b128.summary.json').write_text(json.dumps(record))
    # Withdrawn feature objectives must not request further GPU measurements.
    feature=tmp_path/'feature-feature_ce-lr0.001';feature.mkdir()
    (feature/'complete.json').write_text('{}')
    assert required_devices(tmp_path)=={'card-a','card-b'}


def test_reference_plan_does_not_treat_missing_cells_as_unused_cards(tmp_path):
    with pytest.raises(ValueError,match='all seven completed'):
        required_devices(tmp_path)
