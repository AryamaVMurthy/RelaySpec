"""Dense frozen Qwen3 block-output capture through the pinned vLLM runner."""
import torch
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate


class DensePooler(Pooler):
    def get_supported_tasks(self):
        return {"token_embed"}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate()

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        sizes = cursor.num_scheduled_tokens_cpu.tolist()
        if not DUMMY:
            assert sizes == cursor.prompt_lens_cpu.tolist(), "Unchunked prefill required"
        return [x.contiguous() for x in torch.split(hidden_states, sizes)]


class AUFCaptureQwen3(Qwen3ForCausalLM):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Input-side hooks 2,10,... are block outputs 1,9,..., zero indexed.
        self.model._set_aux_hidden_state_layers((2, 10, 18, 26, 34))
        self.pooler = DensePooler()

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        _, aux = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return torch.cat(aux, dim=-1)


DUMMY = False
INSTALLED = False


def install():
    global INSTALLED
    if INSTALLED:
        return
    INSTALLED = True
    from vllm import ModelRegistry
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    original = GPUModelRunner._dummy_pooler_run

    def dummy(self, *args, **kwargs):
        global DUMMY
        DUMMY = True
        try:
            return original(self, *args, **kwargs)
        finally:
            DUMMY = False

    GPUModelRunner._dummy_pooler_run = dummy
    ModelRegistry.register_model("AUFCaptureQwen3", "experiments.auf_vllm.capture_runtime:AUFCaptureQwen3")
