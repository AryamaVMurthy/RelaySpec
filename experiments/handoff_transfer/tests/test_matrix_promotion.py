from pathlib import Path
import pytest
from experiments.handoff_transfer.matrix.promote import promote
from experiments.handoff_transfer.matrix.task import KINDS


def grid():
    return [dict(family='q8', kind=k, objective=o, lr=lr, steps=100)
            for k in KINDS for o in ('ce', 'auf') for lr in (.0001, .0003, .0006)]


def ranked(runs, stage):
    name = runs[0].name
    kind, objective, _ = name.rsplit('-', 2)
    rates = sorted((float(p.name.split('-lr')[1]) for p in runs), reverse=True)
    return dict(identity=('q8', kind, objective), steps=stage,
                ranked=[dict(lr=lr) for lr in rates])


def test_complete_two_stage_promotion():
    tasks, report = promote(Path('/runs'), grid(), 100, selector=ranked)
    assert len(tasks) == 24 and {t['lr'] for t in tasks} == {.0003, .0006}
    assert {t['steps'] for t in tasks} == {500}
    final, report = promote(Path('/runs'), tasks, 500, selector=ranked)
    assert len(final) == 12 and {t['lr'] for t in final} == {.0006}
    assert {t['steps'] for t in final} == {2000}


@pytest.mark.parametrize('mutation', [lambda g: g[:-1], lambda g: g + [g[0]],
    lambda g: [r for r in g if r['kind'] != 'five_maps']])
def test_partial_or_duplicate_grid_refuses_promotion(mutation):
    with pytest.raises(ValueError):
        promote(Path('/runs'), mutation(grid()), 100, selector=ranked)


def test_failed_verification_cannot_be_skipped():
    def failed(runs, stage):
        raise AssertionError('Checkpoint/export mismatch')
    with pytest.raises(AssertionError, match='Checkpoint/export mismatch'):
        promote(Path('/runs'), grid(), 100, selector=failed)
