"""Audit raw paired request measurements and summarize single-request speed."""
import argparse
import json
from pathlib import Path
import statistics

from experiments.auf_vllm.compare_outputs import compare
from experiments.handoff_transfer.exact_scope.single_request import LABELS,CAMPAIGN


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize_requests(records):
    seconds=sum(r['wall_seconds'] for r in records)
    tokens=sum(r['output_tokens'] for r in records)
    blocks=sum(r['verification_iterations'] for r in records)
    accepted=sum(r['accepted_draft_tokens'] for r in records)
    latency=[r['wall_seconds'] for r in records]
    result=dict(requests=len(records),output_tokens=tokens,total_request_seconds=seconds,
        outputs_at_cap=sum(r['output_tokens']==2048 for r in records),
        length_finishes=sum(r.get('finish_reason')=='length' for r in records),
        output_tps=tokens/seconds,mean_request_seconds=statistics.mean(latency),
        median_request_seconds=statistics.median(latency),
        p95_request_seconds=statistics.quantiles(latency,n=100,method='inclusive')[94],
        draft_blocks=blocks,accepted_draft_tokens=accepted,
        accepted_proposals_per_block=accepted/blocks if blocks else None)
    stages=[]
    for record in records:
        m=record.get('engine_metrics')
        if not isinstance(m,dict):continue
        scheduled,first,last=(m.get(k,0) for k in ('scheduled_ts','first_token_ts','last_token_ts'))
        if 0<scheduled<=first<=last and m.get('first_token_latency',0)>0:
            stages.append(dict(ttft=m['first_token_latency'],prefill_to_first=first-scheduled,
                decode=last-first,tokens_after_first=max(record['output_tokens']-1,0)))
    result['engine_metrics_requests']=len(stages)
    if stages:
        result.update(mean_ttft_seconds=statistics.mean(s['ttft'] for s in stages),
            mean_scheduled_to_first_token_seconds=statistics.mean(s['prefill_to_first'] for s in stages),
            mean_first_to_last_token_seconds=statistics.mean(s['decode'] for s in stages),
            decode_tps=sum(s['tokens_after_first'] for s in stages)/sum(s['decode'] for s in stages)
                if sum(s['decode'] for s in stages)>0 else None)
    return result


def collect(root):
    root=Path(root);results=[];issues=[];progress=[]
    for family in ('q8','cross'):
        cohort=read(root/'exact32e1b8-results'/family/'evaluation/eval.json')
        collected={}
        for label in LABELS:
            mode=label if label in ('ar','native') else 'matrix'
            paths=[];devices=[]
            try:
                for worker in range(4):
                    folder=root/CAMPAIGN/family/f'worker-{worker}'
                    path=folder/label/f'{mode}-r0-w{worker}.jsonl'
                    ar=folder/'ar'/f'ar-r0-w{worker}.jsonl'
                    summary=read(path.with_suffix('.summary.json'))
                    reference=read(ar.with_suffix('.summary.json'))
                    c=summary['contract'];a=reference['contract']
                    for k,v in dict(family=family,count=128,cap=2048,worker_index=worker,workers=4,repeat=0).items():
                        assert c[k]==a[k]==v,(family,label,worker,k)
                    assert c.get('request_batch_size',1)==a.get('request_batch_size',1)==1
                    assert c['manifest_sha256']==a['manifest_sha256']
                    assert c['runtime_config']['model']==a['runtime_config']['model']
                    uuid=summary['gpu_after'][0]['device_uuid']
                    assert all(s[k][0]['device_uuid']==uuid for s in (summary,reference) for k in ('gpu_before','gpu_after'))
                    assert all(check['passed'] for check in summary['asset_checks'])
                    if label=='native':assert c['native_draft_sha256'] and c['native_config_sha256']
                    elif label!='ar':
                        oldlabel='original' if label=='original' else label+'-lr0.0001'
                        expected=read(root/'exact32e1b8-results'/family/oldlabel/'matrix-r0-w0-b128.summary.json')['contract']['export_sha256']
                        assert c['export_sha256']==expected
                    checked=compare([ar],[path])
                    assert checked['count']==checked['exact_matches']==checked['finish_matches']==32
                    actual=rows(path);expected=cohort[worker::4]
                    assert [(r['group_id'],r['prompt_ids']) for r in actual]==[(r['group_id'],r['prompt_token_ids']) for r in expected]
                    paths.append(path);devices.append(uuid)
                records=[r for path in paths for r in rows(path)]
                assert len(records)==len({r['group_id'] for r in records})==128
                collected[label]=dict(family=family,label=label,physical_gpus=devices,
                    exact_ar=128,**summarize_requests(records))
            except (OSError,KeyError,AssertionError,ValueError) as error:
                issues.append(dict(family=family,label=label,error=str(error)))
            progress.append(dict(family=family,label=label,verified_shards=len(paths),required_shards=4))
        for label,row in collected.items():
            for baseline in ('ar','native','original'):
                if baseline in collected:
                    row['speedup_'+baseline]=row['output_tps']/collected[baseline]['output_tps']
            results.append(row)
    return dict(status='complete' if len(results)==14 and not issues else 'incomplete',
        verified_cells=len(results),expected_cells=14,scope='128 requests per target, one active request per GPU, four paired32-question shards, one timing pass and existing fitted checkpoints. TPS is pooled tokens/summed request seconds, not sum of four GPU throughputs.',
        metric_scope='Accepted proposals exclude verifier bonus tokens. Engine scheduled-to-first-token and first-to-last-token intervals are reported only where recorded; these are not isolated GPU-kernel timings.',
        rows=results,progress=progress,issues=issues)


def write_report(report,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    lines=[report['scope'],'',report['metric_scope'],'',
        '| Target group | Method | TPS | Mean request s | / AR | / Native | Accepted proposals/block | Exact AR |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in report['rows']:
        value=lambda k: '-' if row.get(k) is None else f'{row[k]:.3f}'
        lines.append(f"| {row['family']} | {row['label']} | {row['output_tps']:.2f} | {row['mean_request_seconds']:.3f} | {value('speedup_ar')} | {value('speedup_native')} | {value('accepted_proposals_per_block')} | {row['exact_ar']}/128 |")
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    report=collect(a.root);write_report(report,a.out)
    print(json.dumps(dict(status=report['status'],verified_cells=report['verified_cells'],expected_cells=14)))
    if a.require_complete and report['status']!='complete':raise SystemExit(1)
