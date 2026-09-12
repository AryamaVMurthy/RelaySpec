"""Require reference and method measurements on the same physical GPU."""
import json
from experiments.auf_vllm.compare_outputs import compare


def read(path):return json.loads(path.read_text())


def require_same_device(method,reference):
    uuid=method['gpu_after'][0]['device_uuid']
    if any(summary[key][0]['device_uuid']!=uuid for summary in (method,reference) for key in ('gpu_before','gpu_after')):
        raise ValueError('Reference and method used different physical GPUs')
    for key in ('family','manifest_sha256','count','cap','repeat','request_batch_size'):
        if method['contract'][key]!=reference['contract'][key]:
            raise ValueError(f'Reference contract differs: {key}')
    if method['contract']['runtime_config']['model']!=reference['contract']['runtime_config']['model']:
        raise ValueError('Reference target differs')
    if not method['timing_valid'] or not reference['timing_valid']:
        raise ValueError('Profiled measurement is not valid timing')
    return uuid


def audit_matched(root,cells):
    rows=[];issues=[];verified=0
    for cell in cells:
        if not cell['evaluation']:continue
        label=cell['cell'];family,kind,objective=label.split('/')
        summary=cell['training_summary'];lr=summary['contract']['lr'] if kind=='feature' else summary['lr']
        old=root/'exact32e1b8-results'/family
        batch=read(old/'evaluation-batch.json')['request_batch_size']
        case=old/f'{kind}-{objective}-lr{lr}'
        triplet=[]
        try:
            for repeat in range(3):
                stem=f'matrix-r{repeat}-w0-b{batch}'
                method=read(case/f'{stem}.summary.json')
                uuid=method['gpu_after'][0]['device_uuid']
                refs=root/'gpu-matched-references'/family/uuid
                paths={name:refs/name/f'{"ar" if name=="ar" else "matrix"}-r{repeat}-w0-b{batch}.jsonl'
                       for name in ('ar','original','zip')}
                summaries={name:read(path.with_suffix('.summary.json')) for name,path in paths.items()}
                for name,reference in summaries.items():
                    require_same_device(method,reference)
                    canonical=read((old/name/paths[name].name).with_suffix('.summary.json'))
                    if reference['contract']['export_sha256']!=canonical['contract']['export_sha256']:
                        raise ValueError(f'Wrong {name} reference export')
                result=compare([paths['ar']],[case/f'{stem}.jsonl'])
                baselines={name:compare([paths['ar']],[paths[name]]) for name in ('original','zip')}
                for checked in [result,*baselines.values()]:
                    if checked['count']!=checked['exact_matches'] or checked['count']!=checked['finish_matches'] or checked['count']!=128:
                        raise ValueError('GPU-matched output comparison failed')
                counts=method['batch_measurements']
                triplet.append(dict(family=family,method=label,repeat=repeat,device_uuid=uuid,
                    benchmark_job_id=method['job_id'],reference_job_ids={name:s['job_id'] for name,s in summaries.items()},
                    original_tps=baselines['original']['method_tps'],zip_tps=baselines['zip']['method_tps'],
                    speedup_original=result['method_tps']/baselines['original']['method_tps'],
                    speedup_zip=result['method_tps']/baselines['zip']['method_tps'],
                    draft_blocks=sum(b['verification_iterations'] for b in counts),
                    accepted_draft_tokens=sum(b['accepted_draft_tokens'] for b in counts),**result))
            rows.extend(triplet);verified+=1
        except (OSError,KeyError,AssertionError,ValueError) as error:
            issues.append(dict(cell=label,phase='gpu_matched_references',error=str(error)))
    return dict(verified_cells=verified,comparisons=rows,issues=issues)
