"""Opt-in greedy bridge for the installed vLLM V2 DFlash worker.

Experimental: requires a GPU exactness gate before any performance claim.
CPU tokenization/synchronization overhead is part of measured runtime.
"""
import inspect,os
import torch
from transformers import AutoTokenizer
from experiments.handoff_transfer.cross.bridge import TextBridge

def install():
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    if getattr(DFlashSpeculator,'_cross_bridge_installed',False):return
    original=DFlashSpeculator.propose;signature=inspect.signature(original)
    required={'input_batch','num_sampled','num_rejected','last_sampled','next_prefill_tokens','temperature'}
    assert required<=signature.parameters.keys(),'Pinned vLLM proposer API changed'
    def propose(self,*args,**kwargs):
        bound=signature.bind(self,*args,**kwargs);bound.apply_defaults();a=bound.arguments
        if a['dummy_run'] or a['is_profile']:return original(*bound.args,**bound.kwargs)
        if not hasattr(self,'_text_bridge'):
            self._text_bridge=TextBridge(
                AutoTokenizer.from_pretrained(os.environ['CROSS_SOURCE_TOKENIZER'],local_files_only=True),
                AutoTokenizer.from_pretrained(os.environ['CROSS_TARGET_TOKENIZER'],local_files_only=True))
        batch=a['input_batch'];n=batch.num_reqs
        indices=batch.idx_mapping_np[:n].tolist();starts=batch.query_start_loc_np[:n+1].tolist()
        positions=batch.positions[:batch.num_tokens].cpu().tolist();ids=batch.input_ids[:batch.num_tokens].cpu().tolist()
        rejected=a['num_rejected'][:n].cpu().tolist();sampled=a['num_sampled'][:n].cpu().tolist()
        last=a['last_sampled'].clone();prefill=a['next_prefill_tokens'].clone()
        for i,key in enumerate(indices):
            if float(a['temperature'][key])!=0:raise ValueError('Cross bridge currently supports greedy targets only')
            end=starts[i+1]-rejected[i]
            bonus=int(a['last_sampled'][key] if sampled[i]>0 else a['next_prefill_tokens'][key])
            anchor=self._text_bridge.anchor(key,positions[starts[i]:end],ids[starts[i]:end],bonus)
            if sampled[i]>0:last[key]=anchor
            else:prefill[key]=anchor
        a['last_sampled']=last;a['next_prefill_tokens']=prefill
        source_tokens=original(*bound.args,**bound.kwargs)
        converted=[self._text_bridge.proposals(row,self.num_speculative_steps) for row in source_tokens.cpu().tolist()]
        return torch.tensor(converted,device=source_tokens.device,dtype=source_tokens.dtype)
    DFlashSpeculator.propose=propose
    DFlashSpeculator._cross_bridge_installed=True
