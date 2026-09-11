import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest

@pytest.mark.parametrize('objective,expected_weights',[('auf',[0,1,1,0]),('ce',[0,1,1,1])])
def test_selected_objective_supervises_intended_positions(objective,expected_weights):
    path=Path(__file__).resolve().parents[1]/'matrix/objectives.py'
    spec=importlib.util.spec_from_file_location('matrix_objectives_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.DFlashObjectiveTerms=SimpleNamespace
    module.SelectorTerms=SimpleNamespace(zeros=lambda x:SimpleNamespace(**{k:x.new_zeros(()) for k in ['ce_num','probability_num','correct_num','weight_den','covered_num']}))
    model=SimpleNamespace(_matrix_objective=objective,lm_head=torch.nn.Identity(),draft_model=SimpleNamespace(),lk_loss_type=None,_selector_objective_enabled=False)
    logits=torch.tensor([[[[2.,0.],[2.,0.],[0.,2.],[2.,0.]]]],requires_grad=True)
    labels=torch.zeros(1,1,4,dtype=torch.long);mask=torch.tensor([[[0.,1.,1.,1.]]])
    terms=module._dflash_objective_chunk_terms(model,logits,labels,mask,labels)
    loss=terms.ce_loss_num/terms.loss_den
    ce=torch.nn.functional.cross_entropy(logits.reshape(-1,2),labels.reshape(-1),reduction='none')
    weights=torch.tensor(expected_weights,dtype=torch.float)
    torch.testing.assert_close(loss,(ce*weights).sum()/weights.sum())
    loss.backward()
    assert (logits.grad.abs().sum(-1).reshape(-1)>0).tolist()==(weights>0).tolist()
