import ast
from pathlib import Path
import pytest
from experiments.handoff_transfer.seed_source import seed_training_source


def test_seed_transform_preserves_objective_and_budget():
    source = (Path(__file__).resolve().parents[1] / 'port/train.py').read_text()
    for seed in (43, 44):
        changed = seed_training_source(source, seed)
        ast.parse(changed)
        assert f'random.Random({seed}+epoch)' in changed
        assert f'assert a.seed == {seed}' in changed
        assert 'seed{a.seed}' in changed
        assert "torch.manual_seed(a.seed*1000+epoch*10000+rank*3000+num)" in changed
        assert '(loss/2).backward()' in changed and 'processed=step*8' in changed
    with pytest.raises(AssertionError):
        seed_training_source(source.replace('random.Random(42+epoch)', 'random.Random(0)'), 43)
