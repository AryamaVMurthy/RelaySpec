from radical_decode import lookup_continuation


def test_history_lookup_uses_only_observed_continuations():
    assert lookup_continuation([1, 2, 3, 4, 5, 1, 2], 2, 3) == [3, 4, 5]
    assert lookup_continuation([1, 2, 3], 3, 10) == []
    assert lookup_continuation([1, 2, 3, 9, 1, 2, 4, 5, 1, 2], 2, 2) == [4, 5]
    assert lookup_continuation([9, 9, 9], 2, 10) == [9]
