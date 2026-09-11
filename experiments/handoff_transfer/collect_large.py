"""Collect complete fixed-epoch large-data results with explicit compute costs."""
import argparse
import json
from pathlib import Path
from experiments.handoff_transfer.collect_scaling import collect


def main(args):
    jobs=json.loads(args.jobs.read_text())
    assert jobs['epochs']==4
    assert [f['records'] for f in jobs['fits']]==[4096,8192,16384,32768]
    cells=[]
    for fit in jobs['fits']:
        count=fit['records'];updates=fit['updates']
        assert updates==count//2
        root=args.root/f"scaling/q8-n{count}-u{updates}-{fit['job_id']}"
        contract=json.loads((root/'transfer.json').read_text())
        assert contract['records']==count and contract['auf_epochs']==4
        assert contract['auf_record_presentations']==4*count
        result=collect(root,args.root/'q8/measurements')
        assert result['all_exact'] and result['records']==count
        training={}
        for kind in ('fusion_r56','five_maps'):
            summary=json.loads((root/f'modules/full-{kind}/{kind}/summary.json').read_text())
            verified=json.loads((root/f'modules/full-{kind}/{kind}/verification.json').read_text())
            assert summary['optimizer_steps']==verified['steps']==updates
            assert summary['processed_examples']==4*count and verified['status']=='passed'
            assert [e['epoch'] for e in summary['history']]==[1,2,3,4]
            assert all(e['microbatches']==count//2 for e in summary['history'])
            training[kind]=dict(summary,total_adaptation_gpu_hours=summary['training_gpu_hours']+
                               contract['initialization_training_seconds']/3600)
        result['training']=training;cells.append(result)
    report={'status':'complete','all_exact':True,'cells':cells,'auf_epochs':4,
            'compute_note':'Fixed AUF epochs, variable total work. Three-epoch ZIP initialization '
                           'is trained on the same N records and charged as one-GPU time.',
            'scope':'One training seed, three timing repetitions, development evaluation.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('VERIFIED_LARGE_DATA_SWEEP',len(cells),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','jobs','out'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
