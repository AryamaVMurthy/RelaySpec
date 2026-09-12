import pytest
from experiments.handoff_transfer.exact_scope.collect import check_groups


def rows(*keys):return [dict(group_id=key) for key in keys]


def test_disjoint_groups_and_duplicate_or_leaked_groups():
    assert check_groups(rows('a','b'),rows('c'),rows('d'))['exact_group_overlap']==0
    with pytest.raises(ValueError,match='overlap'):check_groups(rows('a'),rows('a'),rows('d'))
    with pytest.raises(ValueError,match='overlap'):check_groups(rows('a'),rows('c'),rows('c'))
    with pytest.raises(ValueError,match='Duplicate'):check_groups(rows('a','a'),rows('c'),rows('d'))
