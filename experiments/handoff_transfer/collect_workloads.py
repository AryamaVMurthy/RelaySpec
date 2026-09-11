"""Collect the four frozen development workloads without selecting subgroups."""
import argparse
from pathlib import Path
from experiments.handoff_transfer.collect_transfer import main as collect


def main(a):
    methods=['ar','normal','zip','handoff-r56','handoff-five']
    if a.family=='q8':methods.append('native')
    for workload in ('math','gsm','code','chat'):
        collect(argparse.Namespace(root=a.root,family=a.family,methods=methods,
            measurements=a.root/f'workloads/{workload}/development',
            out=a.out/f'{a.family}-{workload}.json',
            evaluation=f'128 frozen {workload} development requests, max2048, natural EOS, greedy'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--family',choices=['q8','q14','llama'],required=True)
    main(p.parse_args())
