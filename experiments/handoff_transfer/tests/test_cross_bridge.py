import pytest
from experiments.handoff_transfer.cross.bridge import TextBridge
from experiments.handoff_transfer.tests.test_cross_alignment import Tokenizer

def test_anchor_tracks_committed_prefix_and_rejects_gaps():
    source=Tokenizer(['ab','cd','ef']+list('abcdef'));target=Tokenizer(list('abcdef'))
    target.eos_token_id=99
    bridge=TextBridge(source,target)
    assert bridge.anchor(1,[0],[0],1)==0 # 'ab' source token, not target ID1
    assert bridge.anchor(1,[1,2],[1,2],3)==1 # 'abcd' -> final source 'cd'
    assert bridge.proposals([2],4)==[4,5,99,99]
    with pytest.raises(ValueError,match='Gap'):bridge.anchor(1,[7],[1],2)
    with pytest.raises(ValueError,match='changed'):bridge.anchor(1,[1],[4],2)
