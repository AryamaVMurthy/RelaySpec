"""Summaries of audited complete timing triplets; no fitting-seed inference."""
import csv
import statistics
from pathlib import Path


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
