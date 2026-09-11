from experiments.handoff_transfer.cross.order import epoch_order


def test_epoch_covers_every_record_once_without_crossing_chunk_boundaries():
    a=epoch_order(4096,0);b=epoch_order(4096,1)
    assert sorted(a)==list(range(4096)) and a!=b and a==epoch_order(4096,0)
    for start in range(0,4096,64):
        assert len({i//64 for i in a[start:start+64]})==1
    # Two ranks × two microbatches × two records: exactly one global batch8.
    local=[[a[m*4+r*2+j] for m in range(2) for j in range(2)] for r in range(2)]
    assert len(set(local[0]+local[1]))==8
