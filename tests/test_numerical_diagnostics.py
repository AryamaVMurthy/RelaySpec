import pytest
import torch
from torch import nn

from relayspec.numerical_diagnostics import promote_output_head


class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 4, dtype=torch.bfloat16)
        self.head = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, head):
        self.head = head


def test_head_promotion_preserves_tied_embedding_and_uses_fp32_projection():
    model = TiedModel()
    embedding = model.embedding.weight
    weights = embedding.detach().clone()
    metadata = promote_output_head(model)
    assert metadata["originally_tied"]
    assert model.embedding.weight is embedding
    assert embedding.dtype == torch.bfloat16
    assert torch.equal(embedding, weights)
    assert model.head.weight is not embedding
    hidden = torch.tensor([[0.25, 0.5, -0.125, 0.0625]], dtype=torch.bfloat16)
    actual = model.head(hidden)
    expected = torch.nn.functional.linear(hidden.float(), weights.float())
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_rejects_unsupported_output_head_without_mutating_embeddings():
    model = TiedModel()
    model.head = nn.Linear(4, 3, bias=False, dtype=torch.float32)
    embedding = model.embedding.weight
    with pytest.raises(ValueError, match="BF16 linear"):
        promote_output_head(model)
    assert model.embedding.weight is embedding
