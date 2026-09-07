import pytest
import torch

from relayspec.mapper_campaign import (
    campaign_model_methods,
    campaign_references,
    restore_mapper,
)
from relayspec.relay import TargetFeatureRelay


def test_campaign_shares_model_load_plan_without_losing_candidate_names():
    names = ("native_ar", "native_target_dflash", "relay_small", "relay_large")
    variants = {"relay_small": "small.pt", "relay_large": "large.pt"}
    assert campaign_model_methods(names, variants) == (
        "native_ar",
        "native_target_dflash",
        "relay_p",
    )
    assert names[-1] == "relay_large"
    assert campaign_model_methods(
        ("native_ar", "relay_small", "relay_large"),
        variants,
        relay_method="relay_eagle3",
    ) == ("native_ar", "relay_eagle3")


def test_family_references_preserve_available_controls():
    assert campaign_references(
        "dflash", ["native_ar", "optimized_source_reuse", "relay_a"]
    ) == ["native_ar", "optimized_source_reuse"]
    assert campaign_references(
        "eagle3",
        ["native_ar", "native_target_eagle3", "source_reuse_eagle3", "relay_a"],
    ) == ["native_ar", "native_target_eagle3", "source_reuse_eagle3"]
    with pytest.raises(ValueError, match="requires AR"):
        campaign_references("eagle3", ["relay_a", "source_reuse_eagle3"])
    with pytest.raises(ValueError, match="unsupported"):
        campaign_references("unknown", ["native_ar"])


@pytest.mark.parametrize(
    "names,variants",
    [
        (["relay_p"], {"relay_p": "x.pt"}),
        (["relay_a", "relay_a"], {"relay_a": "x.pt"}),
        (["native_ar"], {"relay_a": "x.pt"}),
        (["other"], {"other": "x.pt"}),
        (["relay_eagle3"], {"relay_eagle3": "x.pt"}),
    ],
)
def test_campaign_rejects_ambiguous_registration(names, variants):
    with pytest.raises(ValueError):
        campaign_model_methods(names, variants)


@pytest.mark.parametrize("kind", ["dense", "factorized", "mlp"])
def test_campaign_restores_each_mapper_and_rejects_wrong_target(kind):
    opts = (
        {"mlp_hidden_width": 3}
        if kind == "mlp"
        else {"factorized_rank": 3}
        if kind == "factorized"
        else {}
    )
    mapper = TargetFeatureRelay(
        target_hidden_size=4, num_taps=3, draft_hidden_size=5, eps=1e-6, **opts
    )
    checkpoint = {"relay": mapper.state_dict(), "target_layer_ids": [1, 3, 5], **opts}
    restored, taps = restore_mapper(
        checkpoint, target_hidden_size=4, draft_hidden_size=5, eps=1e-6
    )
    x = torch.randn(2, 7, 12)
    torch.testing.assert_close(mapper(x), restored(x))
    assert taps == (1, 3, 5)
    with pytest.raises(ValueError, match="input width"):
        restore_mapper(checkpoint, target_hidden_size=6, draft_hidden_size=5, eps=1e-6)


def test_zip_raw_mapper_restore_preserves_feature_scale():
    mapper = TargetFeatureRelay(target_hidden_size=4, num_taps=3,
        draft_hidden_size=5, eps=1e-6, normalize_input=False)
    checkpoint = {'relay': mapper.state_dict(), 'target_layer_ids': [1,3,5],
                  'relay_architecture': 'raw_linear'}
    restored, _ = restore_mapper(checkpoint, target_hidden_size=4,
                                 draft_hidden_size=5, eps=1e-6)
    x = torch.randn(2,7,12)
    torch.testing.assert_close(restored(x), mapper(x))
    torch.testing.assert_close(restored(2*x), 2*restored(x))
