"""Only the frozen-embedding/head, folded linear-context path is supported."""
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
class MapperDFlash(DFlashQwen3ForCausalLM):
 pass
class NormalRelayDFlash(MapperDFlash):
 def combine_hidden_states(self,hidden_states):
  import torch.nn.functional as F
  eps=self.config.relayspec_normal_input_eps
  hidden_states=F.rms_norm(hidden_states,(hidden_states.shape[-1],),eps=eps)
  return super().combine_hidden_states(hidden_states)
_INSTALLED=False
def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 from vllm import ModelRegistry
 from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
 ModelRegistry.register_model('MapperDFlash','mapper_runtime:MapperDFlash')
 ModelRegistry.register_model('NormalRelayDFlash','mapper_runtime:NormalRelayDFlash')
 for name in ['_maybe_share_embeddings','_maybe_share_lm_head']:
  old=getattr(SpecDecodeBaseProposer,name)
  def isolated(self,target,old=old):
   if isinstance(self.model,MapperDFlash):return
   return old(self,target)
  setattr(SpecDecodeBaseProposer,name,isolated)
