import pytest
import torch

from relayspec.training_controls import (
    save_training_checkpoint,
    training_data_accounting,
)


def test_distinct_records_are_separate_from_presentations():
    rows = [{"problem": str(i)} for i in range(8)]
    result = training_data_accounting(rows, {"steps": 16, "distinct_examples": 8}, 4)
    assert result["distinct_records"] == 8
    assert result["record_presentations"] == 64
    with pytest.raises(ValueError, match="distinct budget"):
        training_data_accounting(rows * 2, {"steps": 16, "distinct_examples": 16}, 4)
    with pytest.raises(ValueError, match="equal rank shards"):
        training_data_accounting(rows[:3], {"steps": 16}, 4)


def test_snapshot_preserves_uninterrupted_optimizer_trajectory(tmp_path):
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.randn(5, 3)

    def update(m, opt):
        opt.zero_grad()
        m(inputs).square().mean().backward()
        opt.step()

    update(model, optimizer)
    path = tmp_path / "step.pt"
    rng = torch.get_rng_state().clone()
    save_training_checkpoint(
        path, {"relay": model.state_dict(), "steps": 1}, optimizer, 1.0
    )
    assert torch.equal(rng, torch.get_rng_state())
    saved = torch.load(path, weights_only=True)
    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(saved["relay"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(saved["optimizer"])
    update(model, optimizer)
    update(restored, restored_optimizer)
    for original, reloaded in zip(
        model.parameters(), restored.parameters(), strict=True
    ):
        assert torch.equal(original, reloaded)
    assert all(s["step"] == 2 for s in optimizer.state.values())
    with pytest.raises(FileExistsError):
        save_training_checkpoint(path, {}, optimizer, 2.0)
