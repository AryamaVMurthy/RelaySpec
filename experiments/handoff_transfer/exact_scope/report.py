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
    fields=['family','method','repetitions','request_batch_size','tps_mean','tps_min','tps_max',
            'speedup_ar','speedup_original','speedup_zip']
    with path.with_suffix('.benchmarks.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=[f"Experiment audit: **{audit['status']}**.",
           f"\nVerified fits: {audit['verified_fits']}/30. Fully evaluated trained cells: "
           f"{audit['verified_evaluated_cells']}/30. Transformers confirmations: {audit['transformers_verified']}/2.",
           '\nOnly complete, audited three-repetition comparisons appear below. These are fixed development '
           'requests, one fitting seed, 128 requests per pass, and a 2048-token cap with natural EOS. '
           'Throughput is aggregate batched output tokens/s, not single-request latency. '
           'Ranges describe timing repetitions, not fitting-seed uncertainty. Each speedup is the mean of '
           'ratios paired by timing repetition and model family. Reference passes are reused across jobs '
           'on the same node and GPU model; comparisons can involve different physical GPUs. '
           'The audit retains benchmark job IDs and GPU UUIDs.',
           '\n| Family | Method | Batch | Mean TPS | TPS range | / AR | / Original | / ZIP |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['family']} | {r['method']} | {r['request_batch_size']} | {r['tps_mean']:.1f} | "
                     f"{r['tps_min']:.1f}–{r['tps_max']:.1f} | {r['speedup_ar']:.3f} | "
                     f"{r['speedup_original']:.3f} | {r['speedup_zip']:.3f} |")
    path.with_suffix('.benchmarks.md').write_text('\n'.join(lines)+'\n')
