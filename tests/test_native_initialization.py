import pytest
import torch

from relayspec.native_initialization import initialize_native_columns
from relayspec.relay import TargetFeatureRelay


def test_selected_native_map_equals_teacher_with_other_layers_zeroed():
    teacher = torch.randn(3, 10)
    relay = TargetFeatureRelay(
        target_hidden_size=2,
        num_taps=2,
        draft_hidden_size=3,
        eps=1e-6,
        normalize_input=False,
    )
    record = initialize_native_columns(relay, teacher, [1, 9, 17, 25, 33], [9, 33], 2)
    x = torch.randn(1, 7, 4)
    full = torch.zeros(1, 7, 10)
    full[..., 2:4] = x[..., :2]
    full[..., 8:10] = x[..., 2:]
    torch.testing.assert_close(relay(x), torch.nn.functional.linear(full, teacher))
    assert record["selected_blocks"] == [1, 4]
    relay(x).square().mean().backward()
    assert relay.projection.weight.grad is not None


@pytest.mark.parametrize(
    "kwargs",
    [{"normalize_input": True}, {"normalize_input": False, "mlp_hidden_width": 4}],
)
def test_rejects_interfaces_where_column_copy_is_not_equivalent(kwargs):
    relay = TargetFeatureRelay(
        target_hidden_size=2, num_taps=2, draft_hidden_size=3, eps=1e-6, **kwargs
    )
    with pytest.raises(ValueError, match="unnormalized dense"):
        initialize_native_columns(
            relay, torch.randn(3, 10), [1, 9, 17, 25, 33], [9, 33], 2
        )


def test_rejects_wrong_teacher_width_without_mutating_student():
    relay = TargetFeatureRelay(
        target_hidden_size=2,
        num_taps=2,
        draft_hidden_size=3,
        eps=1e-6,
        normalize_input=False,
    )
    before = relay.projection.weight.detach().clone()
    with pytest.raises(ValueError, match="dimensions"):
        initialize_native_columns(
            relay, torch.randn(3, 8), [1, 9, 17, 25, 33], [9, 33], 2
        )
    torch.testing.assert_close(relay.projection.weight, before)
