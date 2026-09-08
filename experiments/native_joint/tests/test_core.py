import torch
from torch import nn

from core import TAPS, block_view, block_batch, compact_student, prediction_loss, token_ids, cast_parameters


def test_batched_objective_equals_mean_of_individual_values_and_gradients():
    torch.manual_seed(9)
    teacher = torch.randn(3, 15, 19)
    labels = torch.randint(19, (3, 15))
    for kind in ["kl", "ce", "hard_target", "mixed"]:
        for gamma in [0, 7]:
            student = torch.randn(3, 15, 19, requires_grad=True)
            batched, _ = prediction_loss(student, teacher, labels, kind, gamma)
            separate = torch.stack([prediction_loss(student[i], teacher[i], labels[i], kind, gamma)[0] for i in range(3)]).mean()
            assert torch.allclose(batched, separate, atol=1e-6)
            a = torch.autograd.grad(batched, student, retain_graph=True)[0]
            b = torch.autograd.grad(separate, student)[0]
            assert torch.allclose(a, b, atol=1e-7)


def test_variable_prefix_batch_masks_padding_and_preserves_true_positions():
    records = [{"input_ids": torch.arange(40), "taps": torch.randn(5, 40, 2), "final_hidden": torch.randn(40, 3)} for _ in range(2)]
    batch = block_batch(records, [7, 20], [25, 33])
    assert batch["context"].shape == (2, 20, 4)
    assert not batch["attention_mask"][0, :, :, 7:20].any()
    assert batch["attention_mask"][0, :, :, :7].all()
    assert batch["attention_mask"][..., 20:].all()
    assert torch.equal(batch["positions"][0, 20:], torch.arange(7, 23))
    for i, anchor in enumerate([7, 20]):
        view = block_view(records[i], anchor, [25, 33])
        assert torch.equal(batch["context"][i, :anchor], view["context"][0])
        assert torch.equal(batch["labels"][i], view["labels"])
        assert torch.equal(batch["teacher_hidden"][i], view["teacher_hidden"])
    query = torch.randn(2, 1, 16, 4)
    key = torch.randn(2, 1, 36, 4)
    value = torch.randn_like(key)
    reference = torch.nn.functional.scaled_dot_product_attention(query, key, value, batch["attention_mask"])
    key[0, :, 7:20] = 1e4
    value[0, :, 7:20] = -1e4
    perturbed = torch.nn.functional.scaled_dot_product_attention(query, key, value, batch["attention_mask"])
    assert torch.equal(reference, perturbed)


def test_parameter_precision_conversion_preserves_nonpersistent_rope():
    model = nn.Linear(4, 4).bfloat16()
    frequencies = torch.tensor([1., .1, .01, .001])
    model.register_buffer("inv_freq", frequencies.clone(), persistent=False)
    cast_parameters(model, torch.float32)
    cast_parameters(model, torch.bfloat16)
    assert model.weight.dtype == torch.bfloat16
    assert model.inv_freq.dtype == torch.float32
    assert torch.equal(model.inv_freq, frequencies)
    restored = nn.Linear(4, 4).bfloat16()
    restored.register_buffer("inv_freq", frequencies.clone(), persistent=False)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.inv_freq, model.inv_freq)


def test_token_ids_accepts_transformers_batchencoding_and_tensor():
    from transformers import BatchEncoding
    assert token_ids(BatchEncoding({"input_ids": [1, 2, 3]})) == [1, 2, 3]
    assert token_ids({"input_ids": torch.tensor([[1, 2, 3]])}) == [1, 2, 3]


def test_context_excludes_anchor_and_all_future_features():
    record = {"input_ids": torch.arange(40), "taps": torch.arange(5*40*2).reshape(5, 40, 2), "final_hidden": torch.arange(40*3).reshape(40, 3)}
    view = block_view(record, 20, [25, 33])
    assert view["context"].shape == (1, 20, 4)
    assert torch.equal(view["labels"], torch.arange(21, 36))
    assert torch.equal(view["teacher_hidden"], record["final_hidden"][20:35])
    record["taps"][:, 20:] = -999
    record["final_hidden"][35:] = -999
    after = block_view(record, 20, [25, 33])
    assert torch.equal(view["context"], after["context"])
    assert torch.equal(view["teacher_hidden"], after["teacher_hidden"])


def test_distillation_updates_student_not_teacher():
    student = torch.randn(15, 17, requires_grad=True)
    teacher = torch.randn(15, 17, requires_grad=True)
    loss, metrics = prediction_loss(student, teacher, torch.arange(15))
    loss.backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None
    matched, metrics = prediction_loss(teacher.detach(), teacher.detach(), torch.arange(15))
    assert abs(float(matched)) < 1e-6
    assert metrics["teacher_forced_prefix_matches"] == 15


def test_compact_student_copies_native_without_mutating_it():
    from types import SimpleNamespace
    native = nn.Module()
    native.config = SimpleNamespace(hidden_size=2, num_hidden_layers=5, dflash_config={"target_layer_ids": TAPS.copy()})
    native.fc = nn.Linear(10, 2, bias=False)
    native.layers = nn.ModuleList([nn.Module() for _ in range(5)])
    native.target_layer_ids = TAPS.copy()
    for i, layer in enumerate(native.layers):
        layer.self_attn = nn.Module()
        layer.self_attn.layer_idx = i
    student = compact_student(native, [25, 33], [0, 2, 4])
    assert student.fc.in_features == 4
    assert torch.equal(student.fc.weight, native.fc.weight[:, 6:])
    assert [layer.self_attn.layer_idx for layer in student.layers] == [0, 1, 2]
    assert [layer.self_attn.layer_idx for layer in native.layers] == [0, 1, 2, 3, 4]
    assert native.config.dflash_config["target_layer_ids"] == TAPS
    with torch.no_grad():
        student.fc.weight.zero_()
    assert native.fc.weight.abs().sum() > 0
