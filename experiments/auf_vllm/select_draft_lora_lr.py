"""Equal three-rate screens for interface versus drafter adaptation."""
import argparse
import json
import math
from pathlib import Path
from .pilot_data import sha,write


def main(args):
    candidates=[]
    for rate in args.rates:
        path=args.root/f'{args.location}-{args.loss}-lr{rate}/offline-validation.json'
        report=json.loads(path.read_text())
        assert report['records']==1024
        metric=next(x['metrics'] for x in report['results'] if x['epoch']==3)
        assert metric['blocks']==4096 and math.isfinite(metric['accepted_prefix'])
        candidates.append({'lr':rate,'prefix':metric['accepted_prefix'],'sha256':sha(path),
                           'validation':report['manifest_sha256']})
    assert len({x['validation'] for x in candidates})==1
    best=max(x['prefix'] for x in candidates)
    selected=min((x for x in candidates if x['prefix']>=.99*best),key=lambda x:float(x['lr']))
    write(args.out,{'candidates':candidates,'selected_lr':selected['lr'],
                    'rule':'smallest LR within 1% of best epoch3 offline prefix'})
    print(selected['lr'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--location',choices=['draft','fusion','direct'],required=True)
    p.add_argument('--rates',nargs='+',default=['2e-5','1e-4','1e-3'])
    p.add_argument('--loss',choices=['ce','auf'],required=True)
    main(p.parse_args())
