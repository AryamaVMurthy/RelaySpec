import pytest
from experiments.handoff_transfer.matrix.validation import select

def test_tuning_selection_and_both_overlap_checks():
    source=[{'group_id':str(i),'problem':f'Problem {i}'} for i in range(200)]
    evaluation,warmup=select(source,[],source[:128])
    assert len(evaluation)==32 and len(warmup)==4
    with pytest.raises(AssertionError,match='overlap'):select(source,[source[130]],source[:128])
    with pytest.raises(AssertionError,match='Duplicate text'):
        select(source,[{'group_id':'different','problem':'Problem 130'}],source[:128])
