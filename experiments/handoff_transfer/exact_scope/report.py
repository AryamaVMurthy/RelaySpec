"""Summaries of audited complete timing triplets; no fitting-seed inference."""
import csv
import json
import statistics
from pathlib import Path


def objective_contrasts(audit):
    """Compare all matched CE/AUF pairs using acceptance counters, not GPU speed."""
    cells={c['cell']:c for c in audit['cells'] if 'model_contract' in c
           and c['training'] and c['evaluation']}
    results=[]
    for family in ('q8','llama','cross'):
        for architecture in ('five_maps','dense_fusion','five_ba56'):
            names={o:f'{family}/{architecture}/{o}' for o in ('ce','auf')}
            if any(name not in cells for name in names.values()):continue
            ce,auf=(cells[names[o]] for o in ('ce','auf'))
            contracts=[{k:v for k,v in c['model_contract'].items() if k!='objective'} for c in (ce,auf)]
            keys=('seed','lr','optimizer_steps','processed_examples','actual_anchors_rank0',
                  'batch_per_gpu','gradient_accumulation','anchors_per_example','objective_chunk_blocks')
            if (contracts[0]!=contracts[1] or ce['objective_control_transfer_sha256']!=auf['objective_control_transfer_sha256']
                    or any(ce['training_summary'][k]!=auf['training_summary'][k] for k in keys)):
                raise ValueError(f'CE/AUF controls are not matched: {family}/{architecture}')
            grouped={o:sorted((r for r in audit['comparisons'] if r['method']==name),key=lambda r:r['repeat'])
                     for o,name in names.items()}
            for rows in grouped.values():
                if [r['repeat'] for r in rows]!=[0,1,2]:raise ValueError('Incomplete objective comparison')
                for row in rows:
                    if row['count']!=128 or row['exact_matches']!=128 or row['finish_matches']!=128:
                        raise ValueError('Objective contrast lacks exact output verification')
                    if row['draft_blocks']<=0:raise ValueError('Missing draft counters')
            for a,b in zip(grouped['ce'],grouped['auf']):
                for key in ('count','request_batch_size','ar_output_tokens','method_output_tokens'):
                    if a[key]!=b[key]:raise ValueError('Objective workloads are not matched')
            result=dict(family=family,architecture=architecture,fit_seed=ce['training_summary']['seed'],
                fitting_seeds=1,timing_passes=3,requests=128,request_batch_size=grouped['ce'][0]['request_batch_size'])
            for objective,rows in grouped.items():
                result[objective+'_draft_blocks']=[r['draft_blocks'] for r in rows]
                result[objective+'_accepted_tokens']=[r['accepted_draft_tokens'] for r in rows]
                result[objective+'_accepted_per_block']=statistics.mean(r['accepted_draft_tokens']/r['draft_blocks'] for r in rows)
            result['accepted_per_block_gain']=result['auf_accepted_per_block']/result['ce_accepted_per_block']-1
            result['draft_block_reduction']=1-statistics.mean(result['auf_draft_blocks'])/statistics.mean(result['ce_draft_blocks'])
            results.append(result)
    return results


def summarize(audit):
    groups={}
    verified={cell['cell'] for cell in audit['cells'] if cell['evaluation']}
    for row in audit['comparisons']:
        if row['method'] not in ('original','zip') and row['method'] not in verified:
            continue
        key=(row['family'],row['method'])
        repeats=groups.setdefault(key,{})
        if row['repeat'] in repeats:raise ValueError('Duplicate timing repetition')
        if row['count']!=128 or row['exact_matches']!=128 or row['finish_matches']!=128:
            raise ValueError('Unverified output comparison')
        repeats[row['repeat']]=row
    groups={key:value for key,value in groups.items() if set(value)=={0,1,2}}
    results=[]
    for (family,method),repeats in sorted(groups.items()):
        rows=[repeats[i] for i in range(3)]
        batches={row['request_batch_size'] for row in rows}
        if len(batches)!=1:raise ValueError('Serving batch changed across repetitions')
        tps=[row['method_tps'] for row in rows]
        result=dict(family=family,method=method,repetitions=3,request_batch_size=batches.pop(),
                    tps_mean=statistics.mean(tps),tps_min=min(tps),tps_max=max(tps),
                    speedup_ar=statistics.mean(row['throughput_ratio'] for row in rows))
        result['accepted_tokens_per_draft']=(statistics.mean(row['accepted_draft_tokens']/row['draft_blocks'] for row in rows)
                                             if all(row.get('draft_blocks',0)>0 for row in rows) else None)
        for baseline in ('original','zip'):
            reference=groups.get((family,baseline))
            if reference is None:raise ValueError('Complete matched baseline required')
            for i,row in enumerate(rows):
                if (row['request_batch_size'],row['ar_tps'],row['ar_output_tokens']) != (
                        reference[i]['request_batch_size'],reference[i]['ar_tps'],reference[i]['ar_output_tokens']):
                    raise ValueError('Different AR reference or serving batch')
            result['speedup_'+baseline]=statistics.mean(
                row['method_tps']/reference[i]['method_tps'] for i,row in enumerate(rows))
        results.append(result)
    return results


def write_report(audit,path):
    path=Path(path)
    rows=summarize(audit)
    initialization_note=('\nThe retained CE/AUF fits use one token-loss refinement epoch. '
        'Warm-start calibration is separate: '+ '; '.join(
            f"{r['family']} ZIP initialization: {r['records']} records, {r['zip_calibration_epochs']} calibration epochs"
            for r in audit.get('initializers',[])) +
        '. Original-interface CE uses its separate original-MSE initializer. '
        'Refinement timings alone are not total calibration costs. '
        'The six completed feature-loss cells are historical artifacts, outside the primary matrix.'
        if audit.get('initializers') else '')
    fields=['family','method','repetitions','request_batch_size','tps_mean','tps_min','tps_max',
            'speedup_ar','speedup_original','speedup_zip','accepted_tokens_per_draft']
    with path.with_suffix('.benchmarks.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=[f"Experiment audit: **{audit['status']}**.",
           '\n**Provisional references: this table can compare different physical GPUs. '
           'Do not interpret small differences as isolated method gains. Use the GPU-matched report for final comparisons.**',
           f"\nVerified fits: {audit['verified_fits']}/{audit['expected_fits']}. Fully evaluated trained cells: "
           f"{audit['verified_evaluated_cells']}/{audit['expected_fits']}. Transformers confirmations: {audit['transformers_verified']}/2.",
           '\nOnly complete, audited three-repetition comparisons appear below. These are fixed development '
           'requests, one fitting seed, 128 requests per pass, and a 2048-token cap with natural EOS. '
           'Throughput is aggregate batched output tokens/s, not single-request latency. '
           'Ranges describe timing repetitions, not fitting-seed uncertainty. Each speedup is the mean of '
           'ratios paired by timing repetition and model family. Reference passes are reused across jobs '
           'on the same node and GPU model; comparisons can involve different physical GPUs. '
           'The audit retains benchmark job IDs and GPU UUIDs. Accepted/draft is the measured accepted '
           'draft-token count divided by the draft-block count; it excludes the verifier bonus token.',
           initialization_note,
           '\n| Family | Method | Batch | Mean TPS | TPS range | / AR | / Original | / ZIP | Accepted/draft |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        accepted='unavailable' if r['accepted_tokens_per_draft'] is None else f"{r['accepted_tokens_per_draft']:.3f}"
        lines.append(f"| {r['family']} | {r['method']} | {r['request_batch_size']} | {r['tps_mean']:.1f} | "
                     f"{r['tps_min']:.1f}–{r['tps_max']:.1f} | {r['speedup_ar']:.3f} | "
                     f"{r['speedup_original']:.3f} | {r['speedup_zip']:.3f} | {accepted} |")
    path.with_suffix('.benchmarks.md').write_text('\n'.join(lines)+'\n')
    matched={}
    for row in audit.get('gpu_matched_comparisons',[]):
        matched.setdefault((row['family'],row['method']),[]).append(row)
    corrected=[]
    for (family,method),group in sorted(matched.items()):
        assert len(group)==3 and {r['repeat'] for r in group}=={0,1,2}
        corrected.append(dict(family=family,method=method,tps_mean=statistics.mean(r['method_tps'] for r in group),
            ar_tps_mean=statistics.mean(r['ar_tps'] for r in group),
            original_tps_mean=statistics.mean(r['original_tps'] for r in group),
            zip_tps_mean=statistics.mean(r['zip_tps'] for r in group),
            speedup_ar=statistics.mean(r['throughput_ratio'] for r in group),
            speedup_original=statistics.mean(r['speedup_original'] for r in group),
            speedup_zip=statistics.mean(r['speedup_zip'] for r in group)))
    fields=['family','method','tps_mean','ar_tps_mean','original_tps_mean','zip_tps_mean','speedup_ar','speedup_original','speedup_zip']
    with path.with_suffix('.gpu-matched.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(corrected)
    lines=[f"GPU-matched comparisons verified: **{audit.get('gpu_matched_verified_cells',0)}/{audit['expected_primary_fits']}**.",
           '\nEach method is compared with references on the same physical GPU, using the same '
           '128 requests, 2048-token cap, serving batch and repetition index. Three timing repetitions '
           'and one fitting seed. These remain development measurements, not untouched confirmation results.',
           initialization_note,
           '\n| Family | Method | Mean TPS | / AR | / Original | / ZIP |',
           '|---|---|---:|---:|---:|---:|']
    for r in corrected:
        lines.append(f"| {r['family']} | {r['method']} | {r['tps_mean']:.1f} | {r['speedup_ar']:.3f} | "
                     f"{r['speedup_original']:.3f} | {r['speedup_zip']:.3f} |")
    path.with_suffix('.gpu-matched.md').write_text('\n'.join(lines)+'\n')
    contrasts=objective_contrasts(audit)
    scope=('Matched data/initializer/architecture/optimizer settings, one fitting seed per pair. '
           'Three timing passes are not three fitting replications. Accepted proposal tokens per '
           'draft block exclude the verifier bonus; this is not a throughput or universal accuracy guarantee.')
    path.with_suffix('.objective-contrast.json').write_text(json.dumps(dict(scope=scope,rows=contrasts),indent=2)+'\n')
    lines=[scope,'','| Family | Interface | CE accepted/block | AUF accepted/block | Accepted/block gain | Fewer draft blocks |',
           '|---|---|---:|---:|---:|---:|']
    for r in contrasts:
        lines.append(f"| {r['family']} | {r['architecture']} | {r['ce_accepted_per_block']:.4f} | "
                     f"{r['auf_accepted_per_block']:.4f} | {100*r['accepted_per_block_gain']:.2f}% | "
                     f"{100*r['draft_block_reduction']:.2f}% |")
    path.with_suffix('.objective-contrast.md').write_text('\n'.join(lines)+'\n')
