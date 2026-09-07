"""Sparse auxiliary feature capture using vLLM's pooling runner and Qwen3 kernels."""
IS_DUMMY=False
import torch
import os,json
from sampling import key,positions
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate

class SparseContextPooler(Pooler):
 def get_supported_tasks(self):
  return {"token_embed"}
 def get_pooling_updates(self,task):
  return PoolingParamsUpdate()
 def forward(self,hidden_states,pooling_metadata):
  if IS_DUMMY:
   cur=pooling_metadata.get_pooling_cursor()
   return [h if os.environ.get("SD_CAPTURE_DENSE_CHECK")=="1" else h[::4].contiguous() for h in torch.split(hidden_states,cur.num_scheduled_tokens_cpu.tolist())]
  cur=pooling_metadata.get_pooling_cursor()
  sizes=cur.num_scheduled_tokens_cpu.tolist()
  assert sizes==cur.prompt_lens_cpu.tolist(), "Capture requires unchunked prefill"
  outputs=[]
  ids=self.input_ids.detach().cpu().tolist()
  offset=0
  if not hasattr(self,"rows"):
   self.rows=json.load(open(os.environ["SD_CAPTURE_INDEX"]))
  for h in torch.split(hidden_states,sizes):
   # Positions are deterministic and keyed by group ID and prompt/response segment.
   row=self.rows[key(ids[offset:offset+len(h)])]
   selected=list(range(len(h))) if os.environ.get("SD_CAPTURE_DENSE_CHECK")=="1" else positions(len(h),row["boundary"],row["group_id"])
   idx=torch.tensor(selected,device=h.device,dtype=torch.long)
   offset+=len(h)
   outputs.append(h.index_select(0,idx).contiguous())
  return outputs

class CaptureQwen3ForCausalLM(Qwen3ForCausalLM):
 is_pooling_model=True
 def __init__(self,*,vllm_config,prefix=""):
  super().__init__(vllm_config=vllm_config,prefix=prefix)
  self.model._set_aux_hidden_state_layers((2,10,18,26,34))
  self.pooler=SparseContextPooler()
 def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
  self.pooler.input_ids=input_ids
  _,aux=self.model(input_ids,positions,intermediate_tensors,inputs_embeds)
  return torch.cat(aux,dim=-1)

def install():
 from vllm.v1.worker.gpu_model_runner import GPUModelRunner
 old=GPUModelRunner._dummy_pooler_run
 def dummy(self,*args,**kwargs):
  global IS_DUMMY
  IS_DUMMY=True
  try:return old(self,*args,**kwargs)
  finally:IS_DUMMY=False
 GPUModelRunner._dummy_pooler_run=dummy
 from vllm import ModelRegistry
 ModelRegistry.register_model("CaptureQwen3ForCausalLM",
                              "capture_runtime:CaptureQwen3ForCausalLM")
