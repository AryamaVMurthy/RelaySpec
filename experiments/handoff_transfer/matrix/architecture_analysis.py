"""Analytic parameter budgets; no claim about measured effective matrix rank."""
def budgets(source_width,target_width,rank=56,taps=5):
    dense=taps*source_width*target_width
    fusion_ba=rank*(source_width+taps*target_width)
    five_ba=taps*rank*(source_width+target_width)
    matched_rank=round(five_ba/(source_width+taps*target_width))
    return {'source_width':source_width,'target_width':target_width,'taps':taps,
        'dense_fusion_parameters':dense,'five_dense_parameters':dense,
        'fusion_ba_parameters':fusion_ba,'five_ba_parameters':five_ba,
        'fusion_update_rank_upper_bound':min(rank,source_width,taps*target_width),
        'five_ba_update_rank_upper_bound':min(taps*rank,source_width,taps*target_width),
        'approximately_parameter_matched_fusion_rank':matched_rank,
        'matched_fusion_parameters':matched_rank*(source_width+taps*target_width),
        'rank_note':'Upper bounds, not measured fitted ranks'}
