"""Verify that only declared drafter-body LoRA weights change after merging."""
import argparse
from model import *
from peft.tuners.lora.layer import LoraLayer


def main(a):
    torch.manual_seed(42)
    model, cfg = build('draft_lora')
    draft = model.draft_model
    module = R / 'modules' / a.tag / 'draft_lora'
    summary = json.loads((module / 'summary.json').read_text())
    assert summary['optimizer_steps'] == a.steps
    saved = torch.load(module / 'final.pt', weights_only=True, map_location='cuda')
    expected = {k for k, p in draft.named_parameters() if p.requires_grad}
    assert set(saved) == expected and all('.lora_' in k and k.startswith('layers.') for k in saved)
    assert any(not torch.equal(dict(draft.named_parameters())[k], v) for k, v in saved.items())
    draft.load_state_dict(saved, strict=False)
    dest = R / 'exports' / a.tag / 'draft_lora'
    exported = load_file(str(dest / 'model.safetensors'))
    base = load_file(str(Path(TRANSFER['base_export']) / 'model.safetensors'))
    adapted = {name: layer for name, layer in draft.named_modules() if isinstance(layer, LoraLayer)}
    mutable = {name + '.weight' for name in adapted}
    assert set(exported) == set(base)
    assert all(torch.equal(v, exported[k]) for k, v in base.items() if k not in mutable)
    assert any(not torch.equal(base[k], exported[k]) for k in mutable)
    errors = {}
    for name, layer in adapted.items():
        folded = (layer.base_layer.weight.float() + layer.get_delta_weight('default').float()).to(dtype=torch.bfloat16, device='cpu')
        observed = exported[name + '.weight']
        errors[name] = float((folded.float() - observed.float()).square().sum() / folded.float().square().sum().clamp_min(1e-12))
        assert errors[name] < 1e-6
    assert (dest / 'config.json').read_bytes() == (Path(TRANSFER['base_export']) / 'config.json').read_bytes()
    put(module / 'verification.json', {'status': 'passed', 'steps': a.steps,
        'trainable_names': sorted(expected), 'mutable_export_keys': sorted(mutable),
        'frozen_interface_exact': True, 'frozen_non_adapted_exact': True,
        'folded_relative_mse_by_module': errors, 'export_sha256': sha(dest / 'model.safetensors')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--steps', required=True, type=int)
    main(parser.parse_args())
