import torch
import tarfile
import pytest
from safetensors.torch import load_file,save_file
from experiments.handoff_transfer.exact_scope.archive import extract_interface, archive_source_snapshot


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


def test_source_snapshot_preserves_executed_code_without_weights_or_caches(tmp_path):
    source=tmp_path/'source-123'
    (source/'experiments/cross').mkdir(parents=True)
    code=source/'experiments/cross/runtime.py';code.write_text('prefill_anchor = prefill[0]\n')
    (source/'weights.pt').write_bytes(b'not source')
    (source/'.venv').mkdir();(source/'.venv/ignored.py').write_text('ignore')
    target=tmp_path/'code.tar.gz'
    archive_source_snapshot(source,target)
    with tarfile.open(target) as bundle:
        assert bundle.getnames()==['experiments/cross/runtime.py']
        assert bundle.extractfile('experiments/cross/runtime.py').read()==code.read_bytes()
    assert not list(tmp_path.glob('*.tmp'))


def test_source_archive_rejects_missing_or_empty_snapshots(tmp_path):
    with pytest.raises(ValueError,match='No source files'):
        archive_source_snapshot(tmp_path/'missing',tmp_path/'code.tar.gz')
    assert not (tmp_path/'code.tar.gz').exists()
