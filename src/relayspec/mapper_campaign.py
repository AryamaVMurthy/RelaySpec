"""Multiple mapper candidates under one paired decoding measurement protocol."""

from relayspec.relay import TargetFeatureRelay

RESERVED_METHODS = {
    "native_ar",
    "native_target_dflash",
    "naive_source_reuse",
    "optimized_source_reuse",
    "relay_f",
    "relay_p",
    "relay_p_cross_family",
    "direct_slice",
    "frozen_fc_slice",
    "relay_eagle3",
    "source_reuse_eagle3",
    "native_target_eagle3",
}


def campaign_references(family, methods):
    choices = {
        "dflash": ("native_ar", "native_target_dflash", "optimized_source_reuse"),
        "eagle3": ("native_ar", "native_target_eagle3", "source_reuse_eagle3"),
    }
    if family not in choices:
        raise ValueError("unsupported campaign reference family")
    required = {choices[family][0], choices[family][2]}
    if not required.issubset(methods):
        raise ValueError("campaign requires AR and source-reuse references")
    return [name for name in choices[family] if name in methods]


def campaign_model_methods(methods, variants, *, relay_method="relay_p"):
    if relay_method not in {"relay_p", "relay_eagle3"}:
        raise ValueError("unsupported campaign relay family")
    if len(set(methods)) != len(methods):
        raise ValueError("campaign method names must be unique")
    for name, path in variants.items():
        if name in RESERVED_METHODS or not name.startswith("relay_") or not path:
            raise ValueError(
                "variant names must use an unreserved relay_ prefix and a checkpoint"
            )
        if name not in methods:
            raise ValueError("every configured mapper must occur in the method list")
    return tuple(dict.fromkeys(relay_method if n in variants else n for n in methods))


def restore_mapper(checkpoint, *, target_hidden_size, draft_hidden_size, eps):
    taps = tuple(int(i) for i in checkpoint["target_layer_ids"])
    if not taps or tuple(sorted(set(taps))) != taps:
        raise ValueError("checkpoint tap IDs must be sorted and unique")
    architecture = checkpoint.get("relay_architecture", "normalized_linear")
    if architecture not in {"normalized_linear", "scale_preserving_linear"}:
        raise ValueError("unsupported checkpoint input normalization")
    width = checkpoint.get("mlp_hidden_width")
    rank = checkpoint.get("factorized_rank")
    key = (
        "projection.0.weight"
        if width is not None or rank is not None
        else "projection.weight"
    )
    native_width = checkpoint["relay"][key].shape[1]
    if native_width % len(taps):
        raise ValueError("checkpoint input width is not divisible by tap count")
    adapter_width = checkpoint.get("adapter_input_width")
    if (adapter_width or native_width) != target_hidden_size * len(taps):
        raise ValueError("checkpoint input width does not match the target")
    mapper = TargetFeatureRelay(
        target_hidden_size=native_width // len(taps),
        num_taps=len(taps),
        draft_hidden_size=draft_hidden_size,
        eps=eps,
        normalize_input=architecture == "normalized_linear",
        adapter_input_width=adapter_width,
        delta_rank=checkpoint.get("delta_rank"),
        delta_nonlinear=checkpoint.get("delta_nonlinear", False),
        mlp_hidden_width=width,
        factorized_rank=rank,
    )
    mapper.load_state_dict(checkpoint["relay"], strict=True)
    return mapper, taps
