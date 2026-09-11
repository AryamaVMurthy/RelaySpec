import json
import pytest
from experiments.handoff_transfer.matrix.paper_results import build

def test_no_paper_table_from_empty_or_partial_results(tmp_path):
    with pytest.raises(ValueError,match='No complete'):build([],tmp_path/'out')
    partial=tmp_path/'partial.json';partial.write_text(json.dumps({'status':'100_step_fit_verified_validation_pending'}))
    with pytest.raises(AssertionError):build([partial],tmp_path/'out')
    assert not (tmp_path/'out/matrix_table.tex').exists()
