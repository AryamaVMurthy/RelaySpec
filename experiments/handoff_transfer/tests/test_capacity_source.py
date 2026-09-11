import ast
from pathlib import Path
import pytest
from experiments.handoff_transfer.capacity_source import rank_variant


def test_rank56_ablation_preserves_everything_except_variant_name():
    source = (Path(__file__).resolve().parents[1] / 'port/model.py').read_text()
    variant = rank_variant(source, 56)
    expected = source.replace("VARIANTS=['fusion_r56','five_maps']", "VARIANTS=['fusion_capacity']").replace("'fusion_r56'", "'fusion_capacity'")
    assert variant == expected
    for rank in (8, 16, 32, 128, 256):
        tree = ast.parse(rank_variant(source, rank))
        config = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == 'LoraConfig')
        args = {kw.arg: ast.literal_eval(kw.value) for kw in config.keywords}
        assert args['r'] == args['lora_alpha'] == rank
        assert args['lora_dropout'] == 0 and args['bias'] == 'none'
        assert args['target_modules'] == ['fc']
    with pytest.raises(AssertionError):
        rank_variant(source, 0)
