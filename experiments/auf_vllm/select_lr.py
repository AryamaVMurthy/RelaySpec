"""Apply the predeclared one-percent/smallest-LR offline selection rule."""
import argparse
import json
import math
from pathlib import Path
from .pilot_data import sha, write


def select(root):
    result = {}
    for loss in ['zip','ce','auf']:
        candidates = []
        for rate in ['1e-4','3e-4','1e-3']:
            path = root/f'q8-n512-{loss}-lr{rate}-s42/offline-validation.json'
            data = json.loads(path.read_text())
            assert data['records'] == 1024
            final = [x for x in data['results'] if x['epoch'] == 3]
            assert len(final) == 1 and final[0]['metrics']['blocks'] == 4096
            assert math.isfinite(final[0]['metrics']['accepted_prefix']) and final[0]['metrics']['accepted_prefix'] >= 0
            candidates.append({'lr':rate,'accepted_prefix':final[0]['metrics']['accepted_prefix'],
                               'validation_sha256':data['manifest_sha256'],'report_sha256':sha(path)})
        assert len({x['validation_sha256'] for x in candidates}) == 1
        best = max(x['accepted_prefix'] for x in candidates)
        chosen = min((x for x in candidates if x['accepted_prefix'] >= .99*best),key=lambda x:float(x['lr']))
        result[loss] = {'selected_lr':chosen['lr'],'candidates':candidates,'rule':'smallest LR within 1% of best epoch3 offline prefix'}
    assert len({c['validation_sha256'] for x in result.values() for c in x['candidates']}) == 1
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--loss',choices=['zip','ce','auf'])
    args = parser.parse_args()
    result = select(args.root)
    write(args.out,result)
    if args.loss:
        print(result[args.loss]['selected_lr'])
