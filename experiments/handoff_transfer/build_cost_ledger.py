"""Build measured fitting costs; cached-feature preparation is excluded."""
import argparse
import hashlib
import json
from pathlib import Path


def build(reports, family="q8"):
    assert family in ("q8", "q14", "llama")
    sources={}
    def read(name):
        path=reports/name/'summary.json'
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    normal=read(f'{family}-normal-fit');zip_fit=read(f'{family}-zip-fit')
    assert normal['status']=='complete'
    if family=='q8':
        assert zip_fit['status']=='complete'
        config=zip_fit['config']
    else:
        assert zip_fit['status']=='fit_complete_offline_validation_pending'
        config=zip_fit['contract']
        assert config['family']==family and config['objective']=='zip'
    assert normal['contract']['records']==config['records']==4096
    assert normal['contract']['epochs']==config['epochs']==3
    zip_seconds=sum(h['seconds'] for h in zip_fit['history'])
    rows=[{'method':'Normal RelaySpec','trainable_parameters':normal['contract']['trainable_parameters'],
           'initialization_seconds':0,'second_stage_seconds':normal['training_seconds'],
           'sequential_fit_seconds':normal['training_seconds'],'fit_gpu_hours':normal['training_seconds']/3600},
          {'method':'Older ZIP','trainable_parameters':normal['contract']['trainable_parameters'],
           'initialization_seconds':zip_seconds,'second_stage_seconds':0,
           'sequential_fit_seconds':zip_seconds,'fit_gpu_hours':zip_seconds/3600}]
    continuations=[(f'{family}-handoff-r56-fit','ZIP + AUF BA'), (f'{family}-handoff-five-fit','ZIP + AUF five maps')]
    if family=='q8':continuations.append(('q8-ce-fit','ZIP + uniform CE BA'))
    for folder,label in continuations:
        summary=read(folder)
        verified_path=reports/folder/'verification.json'
        verified=json.loads(verified_path.read_text())
        assert verified['status']=='passed' and verified['steps']==summary['optimizer_steps']==2000
        assert summary['world_size']==2 and summary['processed_examples']==16000
        rows.append({'method':label,'trainable_parameters':summary['trainable_parameters'],
                     'initialization_seconds':zip_seconds,'second_stage_seconds':summary['training_seconds'],
                     'sequential_fit_seconds':zip_seconds+summary['training_seconds'],
                     'fit_gpu_hours':zip_seconds/3600+summary['training_gpu_hours']})
        sources[str(verified_path)]=hashlib.sha256(verified_path.read_bytes()).hexdigest()
    return {'status':'complete_fitting_costs_only','family':family,'records':4096,'rows':rows,
            'source_sha256':sources,'exclusions':['rollout generation','source and target feature capture',
                'packing','model loading','diagnostic gates','queue time','decoding evaluation'],
            'interpretation':'Measured fit times from one seed/run. ZIP initialization uses one GPU; '
                'AUF/CE continuation uses two. BA parameter counts describe continuation only; '
                f"the preceding ZIP stage fits {normal['contract']['trainable_parameters']:,} parameters. This is not total adaptation cost."}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--family',choices=('q8','q14','llama'),default='q8')
    args=p.parse_args();result=build(args.reports,args.family)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['rows'],indent=2))
