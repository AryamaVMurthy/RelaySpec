import pytest
from experiments.handoff_transfer.collect_seeds import summarize, METHODS


def test_seed_summary_uses_independent_fit_means():
    reports = {seed: {'rows': {m: {key: value for key in (
        'mean_tps', 'mean_speedup_over_ar', 'mean_tps_ratio_to_normal', 'mean_tps_ratio_to_zip')}
        for m in METHODS}} for seed, value in ((42, 100), (43, 110), (44, 120))}
    result = summarize(reports)['handoff-r56']['mean_tps']
    assert result['mean'] == 110 and result['fitting_seed_stdev'] == 10
    assert result['seed_values'] == [100, 110, 120]
    with pytest.raises(AssertionError):
        summarize({42: reports[42]})
