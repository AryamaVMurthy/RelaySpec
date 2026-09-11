"""Promote complete, verified validation grids; never select on final requests."""
import argparse
import json
from pathlib import Path
from .prepare import digest
from .select import select
from .task import KINDS, values


def promote(root, tasks, stage, selector=select):
    if stage not in (100, 500):
        raise ValueError('Only registered tuning stages may be promoted')
    expected_candidates, keep, next_steps = (3, 2, 500) if stage == 100 else (2, 1, 2000)
    groups = {}
    for task in tasks:
        family, kind, objective, lr, steps = values(task)
        if int(steps) != stage:
            raise ValueError('Mixed fitting budgets in promotion input')
        groups.setdefault((family, kind, objective), []).append(float(lr))
    families = {key[0] for key in groups}
    if len(families) != 1:
        raise ValueError('Promote one complete family grid at a time')
    family = next(iter(families))
    required = {(family, kind, objective) for kind in KINDS for objective in ('ce', 'auf')}
    if set(groups) != required:
        raise ValueError('Incomplete architecture/loss grid')
    promoted, evidence = [], []
    for identity, rates in sorted(groups.items()):
        if len(rates) != expected_candidates or len(set(rates)) != expected_candidates:
            raise ValueError('Missing or duplicate LR candidates')
        if stage == 100 and set(rates) != {0.0001, 0.0003, 0.0006}:
            raise ValueError('Initial LR grid differs from registered protocol')
        family, kind, objective = identity
        runs = [root / family / f'{kind}-{objective}-lr{lr}' for lr in sorted(rates)]
        result = selector(runs, stage)
        if tuple(result['identity']) != identity or result['steps'] != stage:
            raise ValueError('Selection identity differs from task grid')
        ranking = result['ranked']
        if len(ranking) != expected_candidates or {r['lr'] for r in ranking} != set(rates):
            raise ValueError('Selector did not evaluate every candidate')
        for candidate in ranking[:keep]:
            promoted.append(dict(family=family, kind=kind, objective=objective,
                                 lr=candidate['lr'], steps=next_steps))
        evidence.append(result)
    return promoted, {'stage': stage, 'next_steps': next_steps,
        'selection_set': 'separate 32-request validation, cap512',
        'schedule': 'fresh fit from registered initializer at the promoted total schedule',
        'results': evidence}


def main(args):
    tasks = json.loads(args.tasks.read_text())
    promoted, evidence = promote(args.root, tasks, args.stage)
    evidence['input_tasks_sha256'] = digest(args.tasks)
    # Compute/verify every cell before creating a promotable task file.
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'selection.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (args.out / 'tasks.json').write_text(json.dumps(promoted, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--stage', type=int, choices=[100, 500], required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
