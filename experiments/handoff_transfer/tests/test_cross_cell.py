import json
import pytest
from experiments.handoff_transfer.cross.prepare_cell import prepare
from experiments.handoff_transfer.matrix.prepare import digest


def test_cross_cells_are_immutable_and_reject_pilot_initializers(tmp_path):
    init=tmp_path/'init';init.mkdir();index=tmp_path/'index.json';index.write_text('{}')
    transfer=dict(status='complete',family='cross',records=4096,initializer_records=4096,
                  target_adapters=None,full_data_index_sha256=digest(index))
    (init/'transfer.json').write_text(json.dumps(transfer))
    out=tmp_path/'cell'
    a=prepare(init,index,out,'five_ba56','ce',.0001)
    assert a==prepare(init,index,out,'five_ba56','ce',.0001)
    with pytest.raises(AssertionError,match='identity changed'):prepare(init,index,out,'five_ba56','auf',.0001)
    transfer['initializer_records']=512;(init/'transfer.json').write_text(json.dumps(transfer))
    with pytest.raises(AssertionError):prepare(init,index,tmp_path/'pilot','five_ba56','ce',.0001)
