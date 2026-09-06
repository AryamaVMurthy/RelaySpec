"""Capture required Qwen/Llama-style decoder taps without returning all layers."""

import torch


def forward_selected_taps(model, input_ids, layer_ids, **kwargs):
    layers = model.model.layers
    selected = tuple(int(i) for i in layer_ids)
    if not selected or selected != tuple(sorted(set(selected))):
        raise ValueError("Selected taps must be nonempty, sorted and unique")
    if selected[0] < 0 or selected[-1] >= len(layers):
        raise ValueError("Selected tap lies outside target layers")
    if kwargs.get("output_hidden_states"):
        raise ValueError("Selected capture must not request every hidden state")
    captured = {}
    handles = []

    def capture(index):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError("Unsupported decoder-layer output")
            captured[index] = value
        return hook

    try:
        for index in selected:
            # HF hidden_states[-1] is after the final normalization, whereas
            # intermediate entries are the corresponding decoder-layer outputs.
            module = model.model.norm if index == len(layers) - 1 else layers[index]
            handles.append(module.register_forward_hook(capture(index)))
        result = model(input_ids, **{**kwargs, "output_hidden_states": False})
        if set(captured) != set(selected):
            raise ValueError("Target did not execute every selected feature layer")
        return result, tuple(captured[i] for i in selected)
    finally:
        for handle in handles:
            handle.remove()
