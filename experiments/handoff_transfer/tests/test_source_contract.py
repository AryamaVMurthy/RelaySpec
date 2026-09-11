import importlib.util
from pathlib import Path
import pytest
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]

def test_training_port_changes_only_dimensions_and_provenance():
    original=(ROOT/'vendor/code/train.py').read_text()
    expected=original.replace('S,12800','S,INPUT_WIDTH').replace("'initialization':'base draft, zero bias, PEFT random A zero B; seed42'","'initialization':'ZIP epoch-3 initialization; see architecture model contract; seed42'")
    assert (ROOT/'port/train.py').read_text()==expected

@pytest.mark.parametrize("chunk_size", [2, 16])
def test_packaged_auf_chunk_reduction_preserves_loss_and_gradients(chunk_size):
    spec=importlib.util.spec_from_file_location('handoff_objectives_test',ROOT/'vendor/code/objectives.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.DFlashObjectiveTerms=SimpleNamespace
    module.SelectorTerms=SimpleNamespace(zeros=lambda x:SimpleNamespace(**{k:x.new_zeros(()) for k in ['ce_num','probability_num','correct_num','weight_den','covered_num']}))
    model=SimpleNamespace(lm_head=torch.nn.Identity(),draft_model=SimpleNamespace(),lk_loss_type=None,_selector_objective_enabled=False)
    values=torch.zeros(1,33,16,7)
    for b in range(33):
        for j in range(16):values[0,b,j,0 if j<b+2 else 1]=3
    labels=torch.zeros(1,33,16,dtype=torch.long)
    mask=torch.ones_like(labels,dtype=torch.float);mask[...,0]=0;mask[0,3,2]=0
    whole=values.clone().requires_grad_();terms=module._dflash_objective_chunk_terms(model,whole,labels,mask,labels)
    loss=terms.ce_loss_num/terms.loss_den;loss.backward()
    chunked=values.clone().requires_grad_();parts=[]
    for start in range(0,33,chunk_size):
        parts.append(module._dflash_objective_chunk_terms(model,chunked[:,start:start+chunk_size],labels[:,start:start+chunk_size],mask[:,start:start+chunk_size],labels[:,start:start+chunk_size]))
    combined=sum(x.ce_loss_num for x in parts)/sum(x.loss_den for x in parts);combined.backward()
    torch.testing.assert_close(loss,combined)
    torch.testing.assert_close(whole.grad,chunked.grad)
    assert whole.grad[...,0,:].count_nonzero()==0

def test_five_maps_fold_and_gradients_and_frozen_fusion():
    spec=importlib.util.spec_from_file_location('handoff_five_maps_test',ROOT/'port/five_maps.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(7)
    fusion=torch.randn(4,20)
    layer=module.FiveMapProjection(fusion,[torch.randn(4,6) for _ in range(5)])
    x=torch.randn(2,3,30)
    torch.testing.assert_close(layer(x),torch.nn.functional.linear(x,layer.folded()),rtol=1e-5,atol=1e-5)
    layer(x).square().mean().backward()
    assert len(list(layer.parameters()))==5
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in layer.parameters())
    assert layer.fusion.grad is None
    torch.testing.assert_close(layer.fusion,fusion,rtol=0,atol=0)

def test_uniform_ce_control_only_changes_support_and_keeps_later_gradients():
    import types
    source=(ROOT/'vendor/code/objectives.py').read_text()
    ce=types.ModuleType('uniform_ce_test')
    exec(source.replace('loss_weights=weight_mask*support','loss_weights=weight_mask'),ce.__dict__)
    ce.DFlashObjectiveTerms=SimpleNamespace
    ce.SelectorTerms=SimpleNamespace(zeros=lambda x:SimpleNamespace(**{k:x.new_zeros(()) for k in ['ce_num','probability_num','correct_num','weight_den','covered_num']}))
    model=SimpleNamespace(lm_head=torch.nn.Identity(),draft_model=SimpleNamespace(),lk_loss_type=None,_selector_objective_enabled=False)
    logits=torch.tensor([[[[3.,0.],[0.,3.],[3.,0.],[0.,3.]]]],requires_grad=True)
    labels=torch.zeros(1,1,4,dtype=torch.long);mask=torch.tensor([[[0.,1.,1.,1.]]])
    terms=ce._dflash_objective_chunk_terms(model,logits,labels,mask,labels)
    loss=terms.ce_loss_num/terms.loss_den
    expected=torch.nn.functional.cross_entropy(logits.reshape(-1,2)[1:],labels.reshape(-1)[1:])
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert logits.grad[...,0,:].count_nonzero()==0 and (logits.grad[...,2:,:].abs().sum(-1)>0).all()
    original=(ROOT/'port/train.py').read_text()
    expected_source=original.replace("'objective':'AUF'","'objective':'uniform CE'").replace("'objective':'accept-until-fail, first failure included, detached prefix weights'","'objective':'uniform valid-position CE; matched handoff ablation'")
    assert (ROOT/'port_ce/train.py').read_text()==expected_source
