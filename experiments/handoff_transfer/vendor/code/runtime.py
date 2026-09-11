"""All runtime paths enforce the same frozen NanoCoder-LoRA deployment target."""
from common import *
ADAPTER=Path(os.environ['AUF_ADAPTER']).resolve()
def request():
 from vllm.lora.request import LoRARequest
 return LoRARequest(DOMAIN,1,str(ADAPTER))
def llm(spec=False,pool=False,draft=None):
 from vllm import LLM
 import os
 assert os.environ.get('VLLM_BATCH_INVARIANT')=='1'
 kw=dict(disable_log_stats=False,model=str(PAIR/'target'),tokenizer=str(ADAPTER) if DOMAIN=='nanocoder' else str(PAIR/'target'),dtype='bfloat16',enable_lora=True,max_lora_rank=64 if DOMAIN=='kicad' else 32,max_loras=1,max_model_len=16384 if DOMAIN=='kicad' else (8192 if DOMAIN=='nanocoder' else 5120),max_num_seqs=32 if os.environ.get('SD_ROLLOUT')=='1' else 16,max_num_batched_tokens=8192,gpu_memory_utilization=.8,enable_prefix_caching=False,attention_config={'backend':'FLASH_ATTN'},seed=0,generation_config='vllm',async_scheduling=False)
 if pool:kw.update(max_model_len=16384 if DOMAIN=='kicad' else (8192 if DOMAIN=='nanocoder' else 5120),runner='pooling',hf_overrides={'architectures':['CaptureQwen3ForCausalLM']},pooler_config={'task':'token_embed'},enable_chunked_prefill=False,compilation_config={'mode':0})
 else:kw.update(worker_extension_cls='safe_benchmark_worker.SafeWorker',compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64]})
 if spec:kw['speculative_config']={'method':'dflash','model':str(draft or PAIR/'draft'),'num_speculative_tokens':15}
 return LLM(**kw)
