import json
import pytest
from experiments.handoff_transfer.matrix.paper_results import build

def test_no_paper_table_from_empty_or_partial_results(tmp_path):
    with pytest.raises(ValueError,match='No complete'):build([],tmp_path/'out')
    partial=tmp_path/'partial.json';partial.write_text(json.dumps({'status':'100_step_fit_verified_validation_pending'}))
    with pytest.raises(AssertionError):build([partial],tmp_path/'out')
    assert not (tmp_path/'out/matrix_table.tex').exists()

def test_final_report_rejects_missing_registered_cells(tmp_path):
    path=tmp_path/'one.json'
    path.write_text(json.dumps(dict(status='complete_verified',cell=dict(family='q8',kind='five_maps',objective='ce'),
        fit=dict(optimizer_steps=2000,anchors_per_example=512),
        repeats=[dict(count=128,exact_matches=128,finish_matches=128)]*3)))
    with pytest.raises(ValueError,match='Missing 35'):build([path],tmp_path/'out',require_complete=True)
    assert not (tmp_path/'out').exists()
