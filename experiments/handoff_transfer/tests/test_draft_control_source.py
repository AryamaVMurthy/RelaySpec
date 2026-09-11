import ast
from pathlib import Path
import pytest
from experiments.handoff_transfer.draft_control_source import draft_variant, BODY_MODULES


def test_body_control_preserves_objective_and_training_loop():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'port/model.py').read_text()
    for rank in (4, 32):
        transformed = draft_variant(source, rank)
        ast.parse(transformed)
        assert "configure(wrapper,'auf')" in transformed
        assert 'num_anchors=8' in transformed and 'objective_chunk_blocks=2' in transformed
        assert f'target_modules={BODY_MODULES!r}' in transformed
        assert 'not draft.fc.weight.requires_grad' in transformed
        assert transformed[transformed.index('def export('):] == source[source.index('def export('):]
    with pytest.raises(AssertionError, match='Upstream control contract'):
        draft_variant(source.replace('lora_dropout=0', 'lora_dropout=0.1'))
