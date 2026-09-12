import torch
from safetensors.torch import load_file,save_file
from experiments.handoff_transfer.exact_scope.archive import extract_interface


def test_compact_interface_reconstructs_export_without_copying_frozen_model(tmp_path):
    frozen={'other.weight':torch.randn(7,3,dtype=torch.bfloat16)}
    exported={**frozen,'fc.weight':torch.randn(3,5,dtype=torch.bfloat16)}
    source=tmp_path/'full.safetensors';target=tmp_path/'interface.safetensors'
    save_file(exported,source);extract_interface(source,target)
    adapter=load_file(target)
    assert set(adapter)=={'fc.weight'}
    rebuilt={**frozen,**adapter}
    for name,tensor in exported.items():
        assert tensor.dtype==rebuilt[name].dtype and torch.equal(tensor,rebuilt[name])
    assert not list(tmp_path.glob('*.tmp'))
