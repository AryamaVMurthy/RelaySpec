import pytest

from experiments.handoff_transfer.exact_scope.collect_single_request import summarize_requests


def records():
    return [dict(wall_seconds=s,output_tokens=100,verification_iterations=2,
                 accepted_draft_tokens=10,engine_metrics=None) for s in (1.,3.)]


def test_pools_tokens_and_time_instead_of_averaging_tps_or_summing_gpu_rates():
    result=summarize_requests(records())
    assert result['output_tps']==50
    assert result['total_request_seconds']==4
    assert result['mean_request_seconds']==2
    assert result['accepted_proposals_per_block']==5
    assert result['engine_metrics_requests']==0
    assert 'decode_tps' not in result


def test_engine_intervals_use_monotonic_timestamps_and_exclude_first_token():
    data=records()
    for row in data:
        row['engine_metrics']=dict(arrival_time=1e9,scheduled_ts=10.,first_token_ts=10.1,
                                   last_token_ts=11.1,first_token_latency=.2)
    result=summarize_requests(data)
    assert result['mean_scheduled_to_first_token_seconds']==pytest.approx(.1)
    assert result['mean_first_to_last_token_seconds']==1
    assert result['decode_tps']==99
    assert result['mean_ttft_seconds']==.2
