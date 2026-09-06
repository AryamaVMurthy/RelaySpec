import ast
from pathlib import Path

import pytest

from relayspec.pard_adapter import (
    instrument_source,
    trim_pard_tokens,
    verify_pard_decisions,
)


def test_unreviewed_source_rejected():
    with pytest.raises(ValueError, match="reviewed pinned"):
        instrument_source(b"class PardInfer: pass")


def test_trim_retains_eos_and_excludes_overshoot():
    assert trim_pard_tokens([1, 9, 2, 3], max_new_tokens=3, eos_token_id=9) == [1, 9]
    assert trim_pard_tokens([1, 2, 3, 9], max_new_tokens=3, eos_token_id=9) == [1, 2, 3]
    with pytest.raises(ValueError, match="positive"):
        trim_pard_tokens([1], max_new_tokens=0, eos_token_id=9)


@pytest.mark.parametrize("mode", ["none", "detailed", "deferred"])
def test_instrumentation_preserves_every_upstream_statement(mode):
    path = Path("vendor/pard/pard/pard_infer.py")
    if not path.exists():
        pytest.skip("optional pinned PARD source is not installed")
    source = path.read_bytes()
    tree = instrument_source(
        source, trace_targets=mode == "detailed", trace_decisions=mode == "deferred"
    )
    original = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PardInfer"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate"
    )
    loop = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "text"
    )
    assert ast.unparse(loop.body[1]).startswith("_relayspec_request_started =")
    del loop.body[:2]
    boundary = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "output" for t in n.targets)
    )
    assert "self._record_request" in ast.unparse(loop.body[boundary - 1])
    del loop.body[boundary - 3 : boundary]
    if mode != "none":
        inner = next(n for n in loop.body if isinstance(n, ast.While))
        inserted = [
            i
            for i, n in enumerate(inner.body)
            if "self._record_target_" in ast.unparse(n)
        ]
        assert len(inserted) == 1
        del inner.body[inserted[0]]
    assert ast.dump(tree, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


def test_verifier_check_covers_full_blocks_including_raw_overshoot():
    trace = [
        {"output_start": 0, "argmax_ids": list(range(13))},
        {"output_start": 3, "argmax_ids": list(range(20, 33))},
    ]
    verify_pard_decisions([0, 1, 2, 20, 21], [3, 2], trace)
    with pytest.raises(ValueError, match="actual verifier"):
        verify_pard_decisions([0, 1, 2, 20, 99], [3, 2], trace)
    with pytest.raises(ValueError, match="full raw output"):
        verify_pard_decisions([0, 1, 2, 20, 21, 22], [3, 2], trace)
