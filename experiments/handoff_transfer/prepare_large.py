"""Prepare a fully exposed, fixed-epoch large-data AUF comparison."""
import argparse
import json
from pathlib import Path
from experiments.handoff_transfer.pack_transfer import main as pack, sha, write
from experiments.handoff_transfer.prepare_scaling import training_source


def main(args):
    assert args.records in (4096,8192,16384,32768)
    summary=json.loads((args.fit/'summary.json').read_text())
    assert summary['status']=='complete' and summary['config']['records']==args.records
    assert summary['config']['epochs']==3
    assert [h['epoch'] for h in summary['history']]==[1,2,3]
    assert summary['checkpoint_sha256']==sha(args.fit/'resume.pt')
    manifest=args.study/args.data_name/'train.json'
    assert json.loads(manifest.read_text())[:args.records]==json.loads(Path(summary['config']['manifest']).read_text())[:args.records]
    pack(argparse.Namespace(family='q8',records=args.records,study=args.study,
         data_name=args.data_name,out=args.out,base_export=args.fit/'epoch-3/export',prefix=True))
    contract=json.loads((args.out/'transfer.json').read_text())
    contract.update(initialization_training_seconds=sum(h['seconds'] for h in summary['history']),
        initialization_fit_summary_sha256=sha(args.fit/'summary.json'),
        study_axis='four complete AUF epochs; total optimization varies with record count',
        auf_epochs=4,auf_updates=args.records//2,auf_record_presentations=args.records*4)
    write(args.out/'transfer.json',contract)
    (args.out/'scaling_train.py').write_text(training_source())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','study','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--records',type=int,required=True);p.add_argument('--data-name',required=True)
    main(p.parse_args())
