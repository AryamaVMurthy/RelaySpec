import pytest
from experiments.handoff_transfer.matrix.task import values

BASE = dict(family='q8', kind='five_ba56', objective='auf')

def test_existing_and_tuning_tasks():
    assert values(BASE) == ['q8', 'five_ba56', 'auf', '0.0001', '100']
    assert values(dict(BASE, lr=0.0006, steps=2000))[-2:] == ['0.0006', '2000']

@pytest.mark.parametrize('change', [dict(lr=float('nan')), dict(lr=1), dict(steps=True), dict(steps=101), dict(family='unknown'), dict(kind='unknown')])
def test_reject_unregistered_tasks(change):
    with pytest.raises(ValueError):
        values(dict(BASE, **change))
