import torch
import pytest
from experiments.handoff_transfer.cross.paired_batches import batches
from experiments.handoff_transfer.matrix.prepare import digest


def test_unequal_record_lengths_keep_equal_mass_across_batch_boundaries(tmp_path):
    items=[]
    for label,count in [(1,2),(2,5)]:
        path=tmp_path/f'{label}.pt'
        torch.save(dict(group_id=str(label),x=torch.full((count,20480),float(label),dtype=torch.bfloat16),
                        y=torch.zeros(count,12800,dtype=torch.bfloat16)),path)
        items.append(dict(path=str(path),group_id=str(label),positions=count,sha256=digest(path)))
    rows=list(batches(items,batch_size=3))
    assert [len(w) for _,_,w in rows]==[3,3,1]
    mass={label:sum(float(w[x[:,0]==label].sum()) for x,_,w in rows) for label in (1,2)}
    assert mass[1]==pytest.approx(1.) and mass[2]==pytest.approx(1.)
    (tmp_path/'1.pt').write_bytes(b'changed')
    with pytest.raises(AssertionError,match='Paired feature file changed'):list(batches(items,batch_size=3))
