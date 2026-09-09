"""Explicit zero-training controls for unequal-width target and draft features."""

from torch import nn


class UnfittedContext(nn.Module):
    """Slice channels deterministically, optionally using the original draft FC."""

    def __init__(self, *, target_width, draft_width, num_taps, source_fc=None):
        super().__init__()
        if target_width < draft_width or min(draft_width, num_taps) <= 0:
            raise ValueError("slicing requires target width >= positive draft width")
        self.target_width = target_width
        self.draft_width = draft_width
        self.num_taps = num_taps
        self.source_fc = source_fc
        if source_fc is not None:
            if source_fc.in_features != num_taps * draft_width:
                raise ValueError("source FC width does not match selected channels")
            source_fc.requires_grad_(False)

    def forward(self, features):
        if (
            features.ndim != 3
            or features.shape[-1] != self.num_taps * self.target_width
        ):
            raise ValueError("incorrect concatenated target feature width")
        taps = features.reshape(*features.shape[:-1], self.num_taps, self.target_width)
        selected = taps[..., : self.draft_width]
        if self.source_fc is None:
            return selected[..., -1, :]
        return self.source_fc(selected.flatten(-2))
