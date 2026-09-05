"""RelaySpec: relayed cross-scale speculative decoding.

The public neural relay symbols are loaded lazily.  This keeps dataset,
aggregation, and sandboxed code-evaluation utilities usable without importing
the CUDA-enabled PyTorch runtime.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relayspec.relay import TargetFeatureRelay, extract_hidden_taps

__all__ = ["TargetFeatureRelay", "extract_hidden_taps"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from relayspec import relay

        return getattr(relay, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
