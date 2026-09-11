"""Validate array tasks before starting any GPU work."""
import argparse
import json
from pathlib import Path

KINDS = {'normal_ce', 'fusion_r56', 'five_maps', 'dense_fusion', 'five_ba56', 'dense_fresh'}

def values(task):
    family, kind, objective = (task[k] for k in ('family', 'kind', 'objective'))
    if family not in {'q8', 'llama'} or kind not in KINDS or objective not in {'ce', 'auf'}:
        raise ValueError('Unsupported matrix task')
    lr = float(task.get('lr', 0.0001))
    steps = task.get('steps', 100)
    if lr not in {0.0001, 0.0003, 0.0006}:
        raise ValueError('LR outside the registered tuning grid')
    if type(steps) is not int or steps not in {100, 500, 2000}:
        raise ValueError('Unsupported fitting budget')
    return [family, kind, objective, str(lr), str(steps)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('index', type=int)
    args = parser.parse_args()
    if args.index < 0:
        raise ValueError('Negative task index')
    print('\n'.join(values(json.loads(args.manifest.read_text())[args.index])))
