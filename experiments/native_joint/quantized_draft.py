"""Low-bit proposal computation with a separate, untouched BF16 target verifier."""
import copy
import importlib.metadata
import os
from pathlib import Path
import sys
import time

import torch

from run_lane import sha


class DraftHeadTarget:
    """Direct head access drafts; calling the target always uses its original head."""
    def __init__(self, target, draft_head):
        self._target = target
        self.lm_head = draft_head

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __call__(self, *args, **kwargs):
        return self._target(*args, **kwargs)


def attach_shared_linears(modules, linear):
    """Share a compiled functional entry point instead of guarding each bound method."""
    names = []
    def bind(module):
        def forward(inputs):
            return linear(inputs, module.weight, module.bias)
        return forward
    for prefix, model in modules:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = bind(module)
                names.append(f"{prefix}.{name}")
    return names


def build_candidate(native, target, config, output, official, probe, eos):
    started = time.perf_counter()
    sys.path.insert(0, config["package_path"])
    import torchao
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    if importlib.metadata.version("torchao") != "0.15.0" or torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("Quantization dependency pin mismatch")
    student = copy.deepcopy(native)
    draft_head = copy.deepcopy(target.lm_head) if config["quantize_head"] else target.lm_head
    if any(p.data_ptr() == q.data_ptr() for p, q in zip(native.parameters(), student.parameters())):
        raise RuntimeError("Draft copy shares native storage")
    if config["quantize_head"] and draft_head.weight.data_ptr() == target.lm_head.weight.data_ptr():
        raise RuntimeError("Draft head shares target storage")
    proxy = DraftHeadTarget(target, draft_head)
    before = official.dflash_generate(native, target, probe, 64, eos, 0., block_size=16, return_stats=True)
    copied = official.dflash_generate(student, proxy, probe, 64, eos, 0., block_size=16, return_stats=True)
    if not torch.equal(before.output_ids, copied.output_ids) or before.acceptance_lengths != copied.acceptance_lengths:
        raise RuntimeError("Independent draft/head copy changed original decoding")
    recipe = Int4WeightOnlyConfig(group_size=config["group_size"], version=1, set_inductor_config=False)
    quantized = config.get("precision", "int4") == "int4"
    if config.get("precision", "int4") not in {"int4", "bf16"}:
        raise ValueError("Unsupported proposal precision")
    if quantized:
        quantize_(student, recipe)
        if config["quantize_head"]:
            quantize_(draft_head, recipe)
    quantized_modules = {name: type(module.weight).__name__ for name, module in student.named_modules() if isinstance(module, torch.nn.Linear)}
    if quantized and any(kind == "Parameter" for kind in quantized_modules.values()):
        raise RuntimeError("A requested draft linear was not quantized")
    scratch = Path(os.environ["NATIVE_SCRATCH"])/os.environ["SLURM_JOB_ID"]/output.name
    scratch.mkdir(parents=True, exist_ok=False)
    checkpoint = scratch/"quantized-draft.pt"
    torch.save({"student": student.state_dict(), "head": draft_head.state_dict() if config["quantize_head"] else None, "config": config}, checkpoint)
    digest = sha(checkpoint)
    # This is our own just-written file; custom torchao tensor subclasses need
    # deserialization. Never use this path for arbitrary external checkpoints.
    # Tensor subclasses require the indexed device to match their inner storage.
    saved = torch.load(checkpoint, weights_only=False, map_location=f"cuda:{torch.cuda.current_device()}")
    restored = copy.deepcopy(native)
    restored.load_state_dict(saved["student"], assign=True)
    restored_head = copy.deepcopy(target.lm_head) if config["quantize_head"] else target.lm_head
    if config["quantize_head"]:
        restored_head.load_state_dict(saved["head"], assign=True)
    live = official.dflash_generate(student, proxy, probe, 64, eos, 0., block_size=16, return_stats=True)
    reloaded = official.dflash_generate(restored, DraftHeadTarget(target, restored_head), probe, 64, eos, 0., block_size=16, return_stats=True)
    if not torch.equal(live.output_ids, reloaded.output_ids) or live.acceptance_lengths != reloaded.acceptance_lengths:
        raise RuntimeError("Quantized checkpoint failed decode reload gate")
    unchanged = official.dflash_generate(native, target, probe, 64, eos, 0., block_size=16, return_stats=True)
    if not torch.equal(before.output_ids, unchanged.output_ids) or before.acceptance_lengths != unchanged.acceptance_lengths:
        raise RuntimeError("Quantization changed the frozen native/target control")
    compiled_modules = []
    if config.get("compile_linears"):
        torch._dynamo.config.recompile_limit = 32
        shared_linear = torch.compile(torch.nn.functional.linear, dynamic=True, fullgraph=True)
        compiled_modules = attach_shared_linears([("draft", student)]+([("draft_head", draft_head)] if config["quantize_head"] else []), shared_linear)
        compiled = official.dflash_generate(student, proxy, probe, 64, eos, 0., block_size=16, return_stats=True)
        if not torch.equal(live.output_ids, compiled.output_ids) or live.acceptance_lengths != compiled.acceptance_lengths:
            raise RuntimeError("Compiled proposal linear layers changed probe tokens or acceptance")
    return student, proxy, {"config": config, "version": torchao.__version__, "package_init_sha256": sha(torchao.__file__), "quant_api_sha256": sha(Path(torchao.__file__).parent/"quantization/quant_api.py"), "checkpoint": {"path": str(checkpoint), "sha256": digest}, "quantized_modules": quantized_modules, "reload_gate": "pass", "copy_gate": "pass", "frozen_control_gate": "pass", "preparation_seconds": time.perf_counter()-started, "compiled": bool(compiled_modules), "compiled_modules": compiled_modules, "scope": "Proposal precision/linear-compilation experiment. Int4 uses TorchAO0.15 weight-only TensorCoreTiledLayout with groupwise default qparams; BF16 arms are compilation controls. Target weights, target vocabulary head and verification precision remain BF16 and uncompiled. Packing/reload/initial compilation time is separate; any later compilation is included in timed generation."}
