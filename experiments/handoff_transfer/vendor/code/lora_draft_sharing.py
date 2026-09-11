"""Share frozen weights, never target LoRA wrapper/metadata, with DFlash."""
def install():
 from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
 for method,inner in [('_maybe_share_embeddings',True),('_maybe_share_lm_head',False)]:
  original=getattr(SpecDecodeBaseProposer,method)
  def safe(self,target,original=original,inner=inner):
   from vllm.lora.layers.base import BaseLayerWithLoRA
   target_module=target.model.embed_tokens if inner else target.lm_head
   draft_module=self.model.model.embed_tokens if inner else self.model.lm_head
   if isinstance(target_module,BaseLayerWithLoRA):
    assert not isinstance(draft_module,BaseLayerWithLoRA)
    assert draft_module.weight.shape==target_module.weight.shape
    # Valid here: selected adapter has no embedding/head LoRA tensors.
    draft_module.weight=target_module.weight
    print('SD_SAFE_DRAFT_WEIGHT_SHARING', 'embedding' if inner else 'head',type(target_module).__name__,type(draft_module).__name__,flush=True)
    return
   return original(self,target)
  setattr(SpecDecodeBaseProposer,method,safe)
