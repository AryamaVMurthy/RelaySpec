"""Small CPU check of the actual AUF implementation's support and gradients."""
from types import SimpleNamespace
import torch
import objectives as o
from torch.nn import functional as F
o.DFlashObjectiveTerms=SimpleNamespace
o.SelectorTerms=SimpleNamespace(zeros=lambda x:SimpleNamespace(ce_num=x.new_zeros(()),probability_num=x.new_zeros(()),correct_num=x.new_zeros(()),weight_den=x.new_zeros(()),covered_num=x.new_zeros(())))
model=SimpleNamespace(lm_head=torch.nn.Identity(),draft_model=SimpleNamespace(),lk_loss_type='ce',_selector_objective_enabled=False)
cases=[([0,1,0,1],[0,1,1,1],[0,1,1,0]),([0,0,1,1],[0,1,1,1],[0,1,0,0]),([0,1,1,1],[0,1,1,1],[0,1,1,1]),([0,0,1,0],[0,0,1,0],[0,0,1,0])]
for correct,valid,wanted in cases:
 logits=torch.zeros(1,1,4,3)
 for j,c in enumerate(correct):logits[0,0,j,0 if c else 1]=3
 logits.requires_grad_();labels=torch.zeros(1,1,4,dtype=torch.long);mask=torch.tensor(valid,dtype=torch.float32).reshape(1,1,4)
 term=o._dflash_objective_chunk_terms(model,logits,labels,mask,labels)
 ce=F.cross_entropy(logits.reshape(-1,3),labels.reshape(-1),reduction='none');weights=torch.tensor(wanted,dtype=torch.float32)
 expected=(ce*weights).sum()/weights.sum();actual=term.ce_loss_num/term.loss_den
 assert torch.allclose(actual,expected) and term.loss_den.item()==sum(wanted)
 actual.backward();assert (logits.grad[0,0][weights==0]==0).all()
 assert (logits.grad[0,0][weights>0].abs().sum(-1)>0).all()
print('PASS: clean-anchor exclusion, first failure included, later labels excluded, all-correct, invalid positions and zero gradients.')
