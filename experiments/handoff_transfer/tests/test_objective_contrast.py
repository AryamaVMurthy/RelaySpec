import copy
import pytest
from experiments.handoff_transfer.exact_scope.report import objective_contrasts


def fixture():
    cells=[];rows=[]
    for objective,blocks,accepted in [('ce',100,400),('auf',90,410)]:
        name=f'q8/five_maps/{objective}'
        cells.append(dict(cell=name,training=True,evaluation=True,objective_control_transfer_sha256='same data',
            model_contract=dict(objective=objective,variant='five_maps',initialization='same weights'),
            training_summary=dict(seed=42,lr=.0001,optimizer_steps=512,processed_examples=4096,
                actual_anchors_rank0=131072,batch_per_gpu=8,gradient_accumulation=1,
                anchors_per_example=32,objective_chunk_blocks=32)))
        for repeat in range(3):
            rows.append(dict(method=name,repeat=repeat,count=128,exact_matches=128,finish_matches=128,
                request_batch_size=128,ar_output_tokens=500,method_output_tokens=500,
                draft_blocks=blocks,accepted_draft_tokens=accepted))
    return dict(cells=cells,comparisons=rows)


def test_contrast_counts_one_fit_and_retains_all_repeated_counters():
    result=objective_contrasts(fixture())[0]
    assert result['fit_seed']==42 and result['fitting_seeds']==1
    assert result['ce_draft_blocks']==[100]*3 and result['auf_draft_blocks']==[90]*3
    assert result['draft_block_reduction']==pytest.approx(.1)
    assert result['accepted_per_block_gain']==pytest.approx((410/90)/4-1)


@pytest.mark.parametrize('change',['initialization','lr','data'])
def test_mismatched_controls_cannot_be_reported_as_loss_only(change):
    audit=copy.deepcopy(fixture());cell=audit['cells'][1]
    if change=='initialization':cell['model_contract']['initialization']='different weights'
    elif change=='lr':cell['training_summary']['lr']=.0006
    else:cell['objective_control_transfer_sha256']='different data'
    with pytest.raises(ValueError,match='matched'):
        objective_contrasts(audit)


def test_original_reference_metadata_does_not_change_zip_training_data_identity():
    from experiments.handoff_transfer.exact_scope.collect import objective_control_transfer_hash
    original=dict(records=4096,shards={'cache.pt':'abc'},base_sha256='zip')
    extended={**original,'normal_export':'separate-reference','normal_sha256':'normal'}
    assert objective_control_transfer_hash(original)==objective_control_transfer_hash(extended)
    changed={**extended,'shards':{'cache.pt':'different'}}
    assert objective_control_transfer_hash(original)!=objective_control_transfer_hash(changed)
