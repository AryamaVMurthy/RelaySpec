from types import ModuleType,SimpleNamespace
import importlib.util
from pathlib import Path
import torch
from relayspec.losses import relative_interface_mse
from experiments.handoff_transfer.normal_interface import NormalInterface

def test_normal_loss_matches_original_objective_and_only_projection_trains():
    torch.manual_seed(12)
    model=NormalInterface(torch.randn(4,20),torch.randn(4),6,1e-5,1e-6)
    x=torch.randn(8,30);y=torch.randn(8,20)*1e-8
    target=model.normalize(torch.nn.functional.linear(y,model.fusion))
    torch.testing.assert_close(model.feature_loss(x,y).mean(),relative_interface_mse(model(x),target))
    assert [n for n,p in model.named_parameters() if p.requires_grad]==['relay.projection.weight']

def test_runtime_normalization_precedes_projection(monkeypatch):
    class Base(torch.nn.Module):
        def combine_hidden_states(self,x):return torch.nn.functional.linear(x,self.weight)
    fake=ModuleType('vllm.model_executor.models.qwen3_dflash');fake.DFlashQwen3ForCausalLM=Base
    monkeypatch.setitem(__import__('sys').modules,'vllm.model_executor.models.qwen3_dflash',fake)
    path=Path(__file__).resolve().parents[2]/'auf_vllm/runtime_zip/mapper_runtime.py'
    spec=importlib.util.spec_from_file_location('normal_runtime_test',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=mod.NormalRelayDFlash();model.config=SimpleNamespace(relayspec_normal_input_eps=1e-5)
    model.weight=torch.randn(4,30)
    x=torch.randn(8,30)*torch.linspace(.01,10,8)[:,None]
    norm=torch.nn.RMSNorm(30,eps=1e-5,elementwise_affine=False)
    torch.testing.assert_close(model.combine_hidden_states(x),torch.nn.functional.linear(norm(x),model.weight))
    assert not torch.allclose(model.combine_hidden_states(x),torch.nn.functional.linear(x,model.weight))
