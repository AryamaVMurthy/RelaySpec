import sys
import types
import numpy as np
import pytest
import torch
from experiments.handoff_transfer.cross import runtime


@pytest.mark.parametrize('fail', [False, True])
def test_explicit_warmup_scope_clears_and_real_sampling_remains_restricted(monkeypatch, fail):
    names=['vllm','vllm.v1','vllm.v1.worker','vllm.v1.worker.gpu',
           'vllm.v1.worker.gpu.spec_decode','vllm.v1.worker.gpu.spec_decode.dflash',
           'vllm.v1.worker.gpu.spec_decode.dflash.speculator',
           'vllm.v1.worker.gpu_worker','vllm.v1.worker.gpu.warmup']
    modules={name:types.ModuleType(name) for name in names}
    for name,module in modules.items():
        monkeypatch.setitem(sys.modules,name,module)
        if '.' in name:
            parent,leaf=name.rsplit('.',1)
            setattr(modules[parent],leaf,module)
    batch=types.SimpleNamespace(num_reqs=1,num_tokens=1,idx_mapping_np=np.array([0]),
        query_start_loc_np=np.array([0,1]),positions=torch.tensor([0]),input_ids=torch.tensor([1]))
    arguments=dict(input_batch=batch,num_sampled=torch.tensor([1]),num_rejected=torch.tensor([0]),
        last_sampled=torch.tensor([2]),next_prefill_tokens=torch.tensor([0]),temperature=torch.tensor([1.]))
    class Speculator:
        num_speculative_steps=2
        _text_bridge=object() # Non-greedy real input must fail before accessing it.
        def propose(self,input_batch,num_sampled,num_rejected,last_sampled,next_prefill_tokens,
                    temperature,dummy_run=False,is_profile=False):
            return torch.tensor([[7,8]])
    modules[names[6]].DFlashSpeculator=Speculator
    def original_warmup(runner):
        assert runner.speculator._cross_bridge_warmup
        output=runner.speculator.propose(**arguments)
        if fail:raise RuntimeError('synthetic warmup failure')
        return output
    modules['vllm.v1.worker.gpu_worker'].warmup_kernels=original_warmup
    modules['vllm.v1.worker.gpu.warmup'].warmup_kernels=original_warmup
    runtime.install()
    runner=types.SimpleNamespace(speculator=Speculator())
    warmup=modules['vllm.v1.worker.gpu_worker'].warmup_kernels
    if fail:
        with pytest.raises(RuntimeError,match='synthetic warmup failure'):warmup(runner)
    else:
        torch.testing.assert_close(warmup(runner),torch.zeros(1,2,dtype=torch.long))
    assert runner.speculator._cross_bridge_warmup is False
    with pytest.raises(ValueError,match='greedy targets only'):
        runner.speculator.propose(**arguments)
