"""Explicit full-pipeline seed variation; leave the seed42 main recipe intact."""


def seed_training_source(source, seed):
    assert seed in (42, 43, 44)
    old = 'rng=random.Random(42+epoch)'
    assert source.count(old) == 1
    source = source.replace(old, f'rng=random.Random({seed}+epoch)')
    old = "'initialization':'ZIP epoch-3 initialization; see architecture model contract; seed42'"
    assert source.count(old) == 1
    source = source.replace(old, "'initialization':f'ZIP epoch-3 initialization; see architecture model contract; seed{a.seed}'")
    old = 'def main(a):\n rank='
    assert source.count(old) == 1
    return source.replace(old, f'def main(a):\n assert a.seed == {seed}\n rank=')
