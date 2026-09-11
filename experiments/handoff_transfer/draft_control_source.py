"""Change only the trainable location in the handoff model construction."""
BODY_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def draft_variant(source, rank=32):
    assert rank in (4, 32)
    replacements = [
        ("VARIANTS=['fusion_r56','five_maps']", "VARIANTS=['draft_lora']"),
        ("LoraConfig(r=56,lora_alpha=56,lora_dropout=0,bias='none',target_modules=['fc'])",
         f"LoraConfig(r={rank},lora_alpha={rank},lora_dropout=0,bias='none',target_modules={BODY_MODULES!r})"),
        ("assert n==(56*(INPUT_WIDTH+width) if kind=='fusion_r56' else INPUT_WIDTH*width)",
         "from peft.tuners.lora.layer import LoraLayer\n"
         "    adapted = {name: module for name, module in draft.named_modules() if isinstance(module, LoraLayer)}\n"
         "    assert adapted and all(name.startswith('layers.') for name in adapted)\n"
         "    assert not draft.fc.weight.requires_grad and torch.equal(draft.fc.weight, f0)\n"
         f"    assert n == sum({rank} * (m.in_features + m.out_features) for m in adapted.values())"),
        ("'rank':56 if", f"'rank':{rank} if"),
        ("'alpha':56 if", f"'alpha':{rank} if"),
        ("'loss_source_sha256':sha(CODE/'objectives.py'),",
         "'loss_source_sha256':sha(CODE/'objectives.py'),'adapted_modules':sorted(adapted),"
         "'frozen_interface':True,'control':'drafter-body LoRA with matched handoff AUF loop',"),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, f'Upstream control contract changed: {old}'
        source = source.replace(old, new)
    return source.replace("'fusion_r56'", "'draft_lora'")
