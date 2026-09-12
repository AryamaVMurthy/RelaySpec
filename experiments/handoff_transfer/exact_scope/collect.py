"""Audit 21 token fits and six completed historical feature fits and paired128-request evaluations from actual artifacts."""
import argparse,json,sys,hashlib
from pathlib import Path
from experiments.auf_vllm.compare_outputs import compare


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def check_groups(training,evaluation,warmup):
    train={r['group_id'] for r in training};evaluated={r['group_id'] for r in evaluation};warm={r['group_id'] for r in warmup}
    if len(training)!=len(train) or len(evaluation)!=len(evaluated) or len(warmup)!=len(warm):
        raise ValueError('Duplicate group IDs within a split')
    if train&evaluated or train&warm or evaluated&warm:
        raise ValueError('Training, evaluation and warmup groups overlap')
    return dict(training_records=len(train),evaluation_requests=len(evaluated),warmup_requests=len(warm),
                exact_group_overlap=0,scope='Exact group IDs; not a semantic near-duplicate audit')


def audit(root,config_root):
    cells=[];issues=[];comparison_rows=[];reference_hashes={};splits=[];initializers=[]
    for family in ['q8','llama','cross']:
        tasks=read(config_root/f'{family}-tasks.json')
        if family != 'cross': tasks += [dict(kind='feature',objective=o,lr=.001) for o in ['feature_ce','forward_kl','reverse_kl']]
        result_root=root/'exact32e1b8-results'/family
        serving=read(result_root/'evaluation-batch.json') if (result_root/'evaluation-batch.json').exists() else None
        try:
            study=root.parent/'relayspec-auf-20260911'
            sources=([study/'main-q8-n4096/train.json'] if family=='q8' else
                     sorted((study/'main-l3-n4096').glob('part-*/train.json')) if family=='llama' else
                     [root/'cross-full4096/index.json'])
            training=[]
            for source in sources:
                data=read(source);training.extend(data['index'] if family=='cross' else data)
            evaluation=result_root/'evaluation/eval.json';warmup=result_root/'evaluation/warmup.json'
            checked=check_groups(training,read(evaluation),read(warmup))
            assert (checked['training_records'],checked['evaluation_requests'],checked['warmup_requests'])==(4096,128,4)
            splits.append(dict(family=family,**checked,manifests={str(p):sha(p) for p in sources+[evaluation,warmup]}))
        except (OSError,KeyError,AssertionError,ValueError) as error:
            issues.append(dict(cell=family,phase='split_audit',error=str(error)))
        for task in tasks:
            kind,obj,lr=task['kind'],task['objective'],task['lr']
            label=f'{family}/{kind}/{obj}'
            fit=root/'feature-objectives-e1'/family/obj if kind=='feature' else root/'matrix32e1b8'/family/f'{kind}-{obj}-lr{lr}'
            case=result_root/f'{kind}-{obj}-lr{lr}'
            cell=dict(cell=label,training=False,evaluation=False)
            try:
                if kind=='feature':
                    summary=read(fit/'summary.json');contract=read(fit/'contract.json')
                    assert summary['status']=='complete' and contract['epochs']==1 and contract['records']==4096
                    assert contract['objective']==obj and contract['seed']==42
                    assert summary['contract']==contract and summary['export_relative_mse']<1e-6
                    assert len(summary['history'])==1 and summary['history'][0]['epoch']==1
                    assert summary['history'][0]['step']==(contract['positions']+contract['position_batch']-1)//contract['position_batch']
                else:
                    summary=read(fit/f'modules/steps-512/{kind}/summary.json')
                    verification=read(fit/f'modules/steps-512/{kind}/verification.json')
                    assert verification['status']=='passed'
                    assert summary['processed_examples']==4096 and summary['optimizer_steps']==512
                    assert summary['world_size']==1 and summary['batch_per_gpu']==8 and summary['gradient_accumulation']==1
                    assert summary['anchors_per_example']==32 and summary['objective']==obj and summary['seed']==42
                    assert summary['records_rank0']==4096 and 0<summary['actual_anchors_rank0']<=4096*32
                    assert summary['objective_chunk_blocks']==32 and summary['checkpoint']=='final'
                    if family!='cross':
                        assert len(summary['history'])==1 and summary['history'][0]['epoch']==1
                        assert summary['history'][0]['microbatches']==512
                cell['training']=True;cell['training_summary']=summary
            except (OSError,KeyError,AssertionError,ValueError) as error:
                issues.append(dict(cell=label,phase='training',error=str(error)))
            try:
                completion=read(case/'complete.json')
                assert completion['status']=='complete' and completion['requests']==128 and completion['cap']==2048
                assert completion['timing_repetitions']==3 and serving is not None
                batch=serving['request_batch_size'];assert completion['batches']==[batch]
                suffix=f'-b{batch}' if batch>1 else ''
                weights=(fit/'export/model.safetensors') if kind=='feature' else fit/f'exports/steps-512/{kind}/model.safetensors'
                export_hash=sha(weights)
                initial_root=root/'cross-full4096/initializers-e1' if family=='cross' else root/family
                if family not in reference_hashes:
                    transfer=read(initial_root/'transfer.json')
                    reference_hashes[family]={'original':sha(initial_root/'normal/export/model.safetensors'),
                                              'zip':transfer['base_sha256']}
                    initializers.append(dict(family=family,records=transfer['records'],
                        zip_calibration_epochs=transfer['base_training_epochs'],
                        zip_export=transfer['base_export'],zip_export_sha256=transfer['base_sha256'],
                        source=str(initial_root/'transfer.json'),
                        scope='Calibration precedes the one-epoch token-loss refinement; original-interface CE uses its separate original-MSE initializer'))
                expected_exports={**reference_hashes[family],label:export_hash}
                for repeat in range(3):
                    ar=result_root/f'ar/ar-r{repeat}-w0{suffix}.jsonl'
                    eval_summary=read(case/f'matrix-r{repeat}-w0{suffix}.summary.json')
                    assert eval_summary['contract']['export_sha256']==export_hash
                    assert eval_summary['contract']['count']==128 and eval_summary['contract']['cap']==2048
                    for name,folder in [('original',result_root/'original'),('zip',result_root/'zip'),(label,case)]:
                        measured=read(folder/f'matrix-r{repeat}-w0{suffix}.summary.json')
                        assert measured['contract']['export_sha256']==expected_exports[name]
                        assert measured['contract']['repeat']==repeat and measured['contract']['request_batch_size']==batch
                        assert measured['contract']['runtime_config']['max_num_seqs']==batch and measured['timing_valid']
                        result=compare([ar],[folder/f'matrix-r{repeat}-w0{suffix}.jsonl'])
                        assert result['count']==result['exact_matches']==result['finish_matches']==128
                        batches=measured['batch_measurements']
                        comparison_rows.append(dict(family=family,method=name,repeat=repeat,
                                                    benchmark_job_id=measured['job_id'],
                                                    device_uuid=measured['gpu_after'][0]['device_uuid'],
                                                    draft_blocks=sum(b['verification_iterations'] for b in batches),
                                                    accepted_draft_tokens=sum(b['accepted_draft_tokens'] for b in batches),**result))
                cell['evaluation']=True
            except (OSError,KeyError,AssertionError,ValueError) as error:
                issues.append(dict(cell=label,phase='evaluation',error=str(error)))
            cells.append(cell)
    # Deduplicate shared baseline entries from the per-cell audit above.
    rows=list({(row['family'],row['method'],row['repeat']):row for row in comparison_rows}.values())
    from experiments.handoff_transfer.exact_scope.matched import audit_matched
    matched=audit_matched(root,[c for c in cells if '/feature/' not in c['cell']])
    issues.extend(matched['issues'])
    tf=[]
    for mode in ['ce','auf']:
        try:
            folder=root/'transformers-e1b8/q8'
            result=compare([folder/'ar/ar-r0-w0.jsonl'],[folder/f'{mode}/{mode}-r0-w0.jsonl'])
            assert result['count']==result['exact_matches']==result['finish_matches']==128
            tf.append(dict(mode=mode,**result))
        except (OSError,KeyError,AssertionError,ValueError) as error:
            issues.append(dict(cell=f'transformers/{mode}',phase='evaluation',error=str(error)))
    return dict(status='complete' if not issues else 'incomplete',expected_fits=len(cells),expected_primary_fits=21,archived_feature_fits=6,
                verified_fits=sum(c['training'] for c in cells),verified_evaluated_cells=sum(c['evaluation'] for c in cells),
                transformers_verified=len(tf),cells=cells,comparisons=rows,transformers=tf,split_audits=splits,initializers=initializers,
                gpu_matched_verified_cells=matched['verified_cells'],gpu_matched_comparisons=matched['comparisons'],issues=issues)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    result=audit(args.root,Path(__file__).parent)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    from experiments.handoff_transfer.exact_scope.report import write_report
    write_report(result,args.out)
    if args.require_complete and result['status']=='complete':
        from experiments.handoff_transfer.exact_scope.archive import archive
        result['artifact_archive']=archive(result,args.root,args.out.parent/'artifacts',Path(__file__).parent)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cells','comparisons','gpu_matched_comparisons','transformers','split_audits','initializers','issues']}))
    if args.require_complete and result['status']!='complete':sys.exit(1)
