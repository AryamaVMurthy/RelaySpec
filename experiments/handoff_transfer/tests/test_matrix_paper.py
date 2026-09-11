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

def test_complete_report_and_figures_use_all_cells(tmp_path):
    from experiments.handoff_transfer.matrix.figures import render
    from experiments.handoff_transfer.matrix.task import KINDS
    from experiments.handoff_transfer.tests.test_matrix_loss_comparison import result
    paths=[]
    for family in ('q8','llama','cross'):
        for kind in sorted(KINDS):
            for loss in ('ce','auf'):
                # Synthetic fixture only: all outputs live in pytest's tempdir.
                value=result(loss,100 if loss=='ce' else 105)
                value['cell'].update(family=family,kind=kind)
                value.update(mean_paired_ar_speedup=2.,ratio_to_controls={'normal':1.,'zip':1.},
                    control_ratios_by_repeat={'zip':[.99,1.,1.01]},verification={'export_sha256':'fixture'})
                value['fit'].update(training_seconds=100,world_size=2)
                path=tmp_path/f'{family}-{kind}-{loss}.json';path.write_text(json.dumps(value));paths.append(path)
    render(paths,tmp_path/'synthetic-render')
    coverage=json.loads((tmp_path/'synthetic-render/matrix_coverage.json').read_text())
    assert coverage==dict(complete=True,completed_cells=36,required_cells=36,missing=[])
    assert len(json.loads((tmp_path/'synthetic-render/matrix_loss_comparison.json').read_text()))==18
    assert (tmp_path/'synthetic-render/matrix_vs_zip.pdf').stat().st_size>1000
    assert (tmp_path/'synthetic-render/matrix_fit_cost.pdf').stat().st_size>1000
