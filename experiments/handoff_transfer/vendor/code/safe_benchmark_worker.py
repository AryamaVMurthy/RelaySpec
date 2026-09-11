from math_metrics import MetricsWorker
class SafeWorker(MetricsWorker):
 def sd_assert_sharing(self):
  from vllm.lora.layers.base import BaseLayerWithLoRA
  t=self.model_runner.model;d=self.model_runner.drafter.model
  assert not isinstance(d.model.embed_tokens,BaseLayerWithLoRA)
  assert not isinstance(d.lm_head,BaseLayerWithLoRA)
  assert d.model.embed_tokens is not t.model.embed_tokens
  assert d.model.embed_tokens.weight.data_ptr()==t.model.embed_tokens.weight.data_ptr()
  return {'passed':True,'draft_embedding':type(d.model.embed_tokens).__name__,'target_embedding':type(t.model.embed_tokens).__name__,'weight_storage_shared':True,'module_objects_separate':True}
