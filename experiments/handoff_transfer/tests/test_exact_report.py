import pytest
from experiments.handoff_transfer.exact_scope.report import summarize


def example():
    rows=[]
    for method,values in [('original',[200,800,800]),('zip',[200,800,800]),('q8/five_maps/auf',[400,800,1600])]:
        for repeat,(ar,tps) in enumerate(zip([100,200,400],values)):
            rows.append(dict(family='q8',method=method,repeat=repeat,request_batch_size=128,
                             count=128,exact_matches=128,finish_matches=128,
                             ar_tps=ar,method_tps=tps,throughput_ratio=tps/ar,ar_output_tokens=10000))
    return dict(cells=[dict(cell='q8/five_maps/auf',evaluation=True)],comparisons=rows)


def test_ratios_are_paired_by_repetition_not_ratio_of_means():
    row=next(r for r in summarize(example()) if r['method']=='q8/five_maps/auf')
    assert row['speedup_original']==pytest.approx(5/3)
    assert row['speedup_ar']==4


def test_incomplete_or_failed_cell_is_not_reported_as_complete():
    data=example();data['comparisons'].pop()
    assert all(r['method']!='q8/five_maps/auf' for r in summarize(data))
    data=example();data['cells'][0]['evaluation']=False
    assert all(r['method']!='q8/five_maps/auf' for r in summarize(data))


def test_different_ar_references_and_duplicate_repetitions_are_rejected():
    data=example();data['comparisons'][-1]['ar_tps']=401
    with pytest.raises(ValueError,match='Different AR reference'):summarize(data)
    data=example();data['comparisons'].append(data['comparisons'][-1])
    with pytest.raises(ValueError,match='Duplicate timing repetition'):summarize(data)
