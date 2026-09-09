"""Initialize a reduced native input map from the released projection columns."""

import torch


def initialize_native_columns(
    relay, teacher_weight, all_taps, selected_taps, hidden_size
):
    if not isinstance(relay.projection, torch.nn.Linear) or relay.normalize_input:
        raise ValueError(
            "Native column initialization requires an unnormalized dense map"
        )
    if (
        sorted(set(all_taps)) != list(all_taps)
        or sorted(set(selected_taps)) != list(selected_taps)
        or not selected_taps
        or not set(selected_taps).issubset(all_taps)
    ):
        raise ValueError(
            "Native initialization requires sorted unique matching layer IDs"
        )
    weight = relay.projection.weight
    if (
        hidden_size < 1
        or teacher_weight.ndim != 2
        or teacher_weight.shape != (weight.shape[0], hidden_size * len(all_taps))
        or weight.shape[1] != hidden_size * len(selected_taps)
    ):
        raise ValueError("Native teacher and selected interface dimensions differ")
    if not torch.isfinite(teacher_weight).all():
        raise ValueError("Native teacher contains nonfinite weights")
    blocks = [list(all_taps).index(layer) for layer in selected_taps]
    selected = torch.cat(
        [teacher_weight[:, b * hidden_size : (b + 1) * hidden_size] for b in blocks],
        dim=1,
    )
    with torch.no_grad():
        weight.copy_(selected)
    return dict(
        mode="native_columns",
        selected_blocks=blocks,
        selected_taps=list(selected_taps),
        initial_weight_norm=float(weight.detach().float().norm()),
        scope="Selected released projection columns only. Other-layer contributions are absent at initialization. Subsequent fitting updates all mapper weights.",
    )
