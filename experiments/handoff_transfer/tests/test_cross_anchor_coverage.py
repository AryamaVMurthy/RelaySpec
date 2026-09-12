import pytest
from experiments.handoff_transfer.cross.assemble import anchor_coverage

def test_coverage_does_not_inflate_short_records():
    result=anchor_coverage([16,512,1024],limit=512)
    assert result['records_below_limit']==1
    assert result['distinct_anchors_available_per_complete_epoch']==1040
    assert result['median_eligible']==512
    with pytest.raises(AssertionError):anchor_coverage([0,512])


def test_current_study_coverage_caps_each_record_at32():
    result=anchor_coverage([16,512,1024])
    assert result['anchor_limit']==32
    assert result['records_below_limit']==1
    assert result['distinct_anchors_available_per_complete_epoch']==80
