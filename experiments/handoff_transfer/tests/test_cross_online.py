import torch
from experiments.handoff_transfer.cross.online import sample_aligned_anchors,gather_source_blocks

def test_nonadjacent_target_anchors_and_unequal_record_lengths():
    valid=torch.tensor([[0,1,0,1,0,1],[0,0,1,0,0,0]],dtype=torch.bool)
    anchors,keep=sample_aligned_anchors(valid,512)
    assert anchors[0].tolist()==[1,3,5]
    assert anchors[1,keep[1]].tolist()==[2]
    assert keep.sum().item()==4
    blocks=torch.arange(2*6*4).reshape(2,6,4)
    labels,weights=gather_source_blocks(blocks,torch.ones_like(blocks),anchors,keep)
    assert labels[0,1].tolist()==blocks[0,3].tolist()
    assert weights[:,:,0].sum()==0
    assert weights[1,1:].sum()==0
    assert weights.sum()==12

def test_collate_never_uses_target_ids_as_source_labels():
    from experiments.handoff_transfer.cross.batch import collate
    record={'target_ids':[900,901,902,903], 'features':torch.randn(4,10),
        'alignment':{'blocks':[{'target_anchor':2,'context_exclusive_end':2,'source_labels':[7,8,9]}]}}
    ids,hidden,eligible,labels,valid=collate([record],block_size=4)
    assert ids[0,2].item()==7
    assert labels[0,2].tolist()==[7,8,9,0]
    assert valid[0,2].tolist()==[True,True,True,False]
    assert eligible.tolist()==[[0.,0.,1.,0.]]
    torch.testing.assert_close(hidden[0],record['features'].to(torch.bfloat16))
