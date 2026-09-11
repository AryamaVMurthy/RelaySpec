from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from experiments.auf_vllm.runtime_zip.metrics import MetricsWorker


def test_rejects_stale_context_kv_even_when_layer_weights_match(tmp_path):
    torch.manual_seed(42)
    tensors={}
    sizes={'self_attn.q_proj':(4,3),'self_attn.k_proj':(2,3),'self_attn.v_proj':(2,3),
           'self_attn.o_proj':(3,4),'mlp.gate_proj':(5,3),'mlp.up_proj':(5,3),'mlp.down_proj':(3,5)}
    for name,shape in sizes.items():tensors['layers.0.'+name+'.weight']=torch.randn(shape)
    save_file(tensors,str(tmp_path/'model.safetensors'))
    get=lambda name:tensors['layers.0.'+name+'.weight']
    weight=lambda x:SimpleNamespace(weight=x)
    layer=SimpleNamespace(self_attn=SimpleNamespace(qkv_proj=weight(torch.cat([get('self_attn.'+n) for n in ['q_proj','k_proj','v_proj']])),
                                o_proj=weight(get('self_attn.o_proj'))),
          mlp=SimpleNamespace(gate_up_proj=weight(torch.cat([get('mlp.'+n) for n in ['gate_proj','up_proj']])),
                              down_proj=weight(get('mlp.down_proj'))))
    model=SimpleNamespace(layers=[layer],_fused_kv_weight=torch.cat([get('self_attn.k_proj'),get('self_attn.v_proj')]))
    worker=MetricsWorker();worker.model_runner=SimpleNamespace(get_draft_model=lambda:SimpleNamespace(model=model))
    assert worker.sd_verify_draft_projections(tmp_path)['fused_context_kv_verified']
    model._fused_kv_weight[0,0]+=1
    with pytest.raises(AssertionError):worker.sd_verify_draft_projections(tmp_path)
