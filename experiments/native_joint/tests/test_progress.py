import torch

from core import ProgressHead, progress_loss, stop_aware_prefix_mask


def test_progress_head_starts_as_exact_identity():
    hidden = torch.randn(2, 15, 8)
    head = ProgressHead(8, 2)
    assert torch.equal(hidden, head(hidden))


def test_invalid_post_rejection_labels_have_zero_gradient():
    logits = torch.randn(1, 5, 7, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    native = torch.tensor([[1, 2, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, False, False]])
    loss, metrics = progress_loss(logits, labels, valid, native, error_weight=4.)
    loss.backward()
    assert logits.grad[:, :3].abs().sum() > 0
    assert logits.grad[:, 3:].abs().sum() == 0
    assert metrics["first_errors"] == 1
    changed = labels.clone()
    changed[:, 3:] = 0
    other, _ = progress_loss(logits.detach(), changed, valid, native, error_weight=4.)
    assert torch.equal(loss.detach(), other)


def test_position_specific_correction_preserves_other_positions_exactly():
    hidden = torch.ones(2, 15, 8)
    head = ProgressHead(8, 2, positions=[0])
    with torch.no_grad():
        head.up.weight.fill_(1)
        head.down.weight.fill_(1)
    output = head(hidden)
    assert torch.equal(output[:, 1:], hidden[:, 1:])
    assert not torch.equal(output[:, 0], hidden[:, 0])


def test_margin_preservation_loss_detaches_teacher_and_masks_future():
    student = torch.randn(1, 5, 7, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    native = torch.tensor([[1, 2, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, False, False]])
    loss, _ = progress_loss(student, labels, valid, native, kind="margin_kl", native_logits=teacher)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[:, 3:].abs().sum() == 0


def test_eos_label_is_kept_but_later_positions_are_masked():
    native = torch.tensor([[1, 99, 2, 3], [1, 2, 3, 4]])
    valid = torch.ones_like(native, dtype=torch.bool)
    assert stop_aware_prefix_mask(valid, native, [99]).tolist() == [[True, True, False, False], [True, True, True, True]]


def test_soft_progress_rewards_extending_prefix_and_masks_unverified_tail():
    logits = torch.tensor([[[5., 0.], [0., 1.], [1., 0.]]], requires_grad=True)
    labels = torch.zeros(1, 3, dtype=torch.long)
    valid = torch.tensor([[True, True, False]])
    native = torch.tensor([[0, 1, 0]])
    loss, _ = progress_loss(logits, labels, valid, native, kind="soft_progress")
    loss.backward()
    assert logits.grad[:, 2].abs().sum() == 0
    corrected = logits.detach().clone()
    corrected[:, 1, 0] = 3.
    better, _ = progress_loss(corrected, labels, valid, native, kind="soft_progress")
    assert better < loss.detach()


def test_warm_hints_preserve_known_token_and_unused_masks():
    from radical_decode import warm_start_noise
    embedding = torch.nn.Embedding(10, 4)
    noise = embedding(torch.tensor([[2, 9, 9, 9, 9]]))
    assert torch.equal(warm_start_noise(noise, embedding, [3, 4], 0.), noise)
    blended = warm_start_noise(noise, embedding, [3, 4], .25)
    assert torch.equal(blended[:, :1], noise[:, :1])
    assert torch.equal(blended[:, 3:], noise[:, 3:])
    assert torch.allclose(blended[:, 1:3], .75*noise[:, 1:3]+.25*embedding(torch.tensor([[3, 4]])))


def test_rejected_tail_excludes_corrective_token_and_preserves_alignment():
    from radical_decode import rejected_tail
    proposal = torch.tensor([10, 11, 12, 13, 14])
    posterior = torch.tensor([11, 99, 88, 77, 66])
    # One draft token matches. The next known token is posterior[1] == 99.
    assert rejected_tail(proposal, posterior, 1, "draft") == [13, 14]
    assert rejected_tail(proposal, posterior, 1, "target") == [88, 77, 66]
    assert rejected_tail(proposal, posterior, 4, "target") == []


def test_lazy_verification_preserves_first_rejection_and_corrective_token():
    from radical_decode import lazy_verify_tokens
    proposal = torch.tensor([[0, 1, 2, 3, 4]])
    for mismatch in range(5):
        predictions = torch.tensor([[1, 2, 3, 4, 5]])
        if mismatch < 4:
            predictions[0, mismatch] = 6
        logits = torch.nn.functional.one_hot(predictions, 7).float()
        for chunk in [1, 2, 3, 5]:
            visits = []
            def head(hidden):
                visits.append(hidden.shape[1])
                return hidden
            posterior, accepted = lazy_verify_tokens(proposal, logits[:, :chunk], logits, head, chunk)
            expected = min(mismatch, 4)
            assert accepted.item() == expected
            assert posterior[0, expected] == predictions[0, expected]
            assert posterior.shape[1] == min(5, ((expected//chunk)+1)*chunk)
            assert sum(visits)+min(chunk, 5) == posterior.shape[1]


def test_draft_head_proxy_never_changes_target_verification_head():
    from quantized_draft import DraftHeadTarget
    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 3, bias=False)
        def forward(self, x):
            return self.lm_head(x)
    target = Target()
    draft_head = torch.nn.Linear(2, 3, bias=False)
    proxy = DraftHeadTarget(target, draft_head)
    x = torch.ones(1, 2)
    assert torch.equal(proxy(x), target(x))
    assert torch.equal(proxy.lm_head(x), draft_head(x))
    assert proxy._target.lm_head is target.lm_head
    assert proxy.lm_head is not target.lm_head


def test_shared_compiled_linear_handles_more_than_32_distinct_modules():
    from quantized_draft import attach_shared_linears
    modules = torch.nn.ModuleList([torch.nn.Linear(8, 8).requires_grad_(False) for _ in range(40)])
    x = torch.randn(1, 5, 8)
    expected = [module(x) for module in modules]
    shared = torch.compile(torch.nn.functional.linear, backend="eager", dynamic=True, fullgraph=True)
    names = attach_shared_linears([("test", modules)], shared)
    assert len(names) == 40
    assert all(torch.equal(module(x), reference) for module, reference in zip(modules, expected))
