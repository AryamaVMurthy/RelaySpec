from experiments.handoff_transfer.matrix.architecture_analysis import budgets

def test_requested_parameter_budgets():
    q=budgets(2560,4096);l=budgets(4096,3072)
    assert q['dense_fusion_parameters']==q['five_dense_parameters']==52428800
    assert q['fusion_ba_parameters']==1290240 and q['five_ba_parameters']==1863680
    assert l['dense_fusion_parameters']==62914560
    assert l['fusion_ba_parameters']==1089536 and l['five_ba_parameters']==2007040
    assert q['approximately_parameter_matched_fusion_rank']==81
    assert l['approximately_parameter_matched_fusion_rank']==103
    assert q['fusion_update_rank_upper_bound']==56 and q['five_ba_update_rank_upper_bound']==280
