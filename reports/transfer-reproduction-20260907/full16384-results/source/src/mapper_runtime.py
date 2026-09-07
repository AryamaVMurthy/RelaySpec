"""Only the frozen-embedding/head, folded linear-context path is supported."""
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
class MapperDFlash(DFlashQwen3ForCausalLM):
 pass
_INSTALLED=False
def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 from vllm import ModelRegistry
 from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
 ModelRegistry.register_model('MapperDFlash','mapper_runtime:MapperDFlash')
 for name in ['_maybe_share_embeddings','_maybe_share_lm_head']:
  old=getattr(SpecDecodeBaseProposer,name)
  def isolated(self,target,old=old):
   if isinstance(self.model,MapperDFlash):return
   return old(self,target)
  setattr(SpecDecodeBaseProposer,name,isolated)
