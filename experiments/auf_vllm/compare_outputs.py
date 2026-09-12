"""Compare complete token sequences and aggregate paired decoding evidence."""
import argparse
import json
import math
from pathlib import Path
from .pilot_data import write


def compare(ar, method):
    def read(paths):
        result={}
        caps=set(); batch_sizes=set()
        for path in paths:
            assert path.with_suffix('.summary.json').exists(),path
            summary=json.loads(path.with_suffix('.summary.json').read_text())
            assert summary['timing_valid']
            contract=summary.get('contract',summary.get('args',{}))
            caps.add(contract['cap'])
            batch_size=contract.get('request_batch_size',1);batch_sizes.add(batch_size)
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            if batch_size>1:
                assert 2<=batch_size<=128 and contract['workers']==1
                batches=summary['batch_measurements']
                assert len({b['batch_id'] for b in batches})==len(batches)
                assert {r['batch_id'] for r in rows}=={b['batch_id'] for b in batches}
                for batch in batches:
                    group=[r for r in rows if r['batch_id']==batch['batch_id']]
                    assert len(group)==batch['requests'] and 0<len(group)<=batch_size
                    assert all(r['batch_requests']==len(group) for r in group)
                    assert sum(r['output_tokens'] for r in group)==batch['output_tokens']
                    assert batch['wall_seconds']>0
                    assert math.isclose(sum(r['wall_seconds'] for r in group),batch['wall_seconds'])
            else:
                assert all('batch_id' not in r for r in rows), 'Batched rows lack a batch contract'
            for row in rows:
                assert row['group_id'] not in result,'duplicate request'
                assert row['timing_valid']
                assert row['output_tokens']==len(row['output_ids']) and 0<row['output_tokens']<=contract['cap'], 'Invalid token accounting'
                assert math.isfinite(row['wall_seconds']) and row['wall_seconds']>0, 'Invalid measured wall time'
                result[row['group_id']]=row
        assert len(caps)==1
        assert len(batch_sizes)==1
        return result,caps,batch_sizes
    (a,caps_a,batches_a),(b,caps_b,batches_b)=read(ar),read(method)
    assert batches_a==batches_b, 'Unequal serving batch sizes'
    batch_size=next(iter(batches_a))
    assert caps_a==caps_b,'unequal output caps'
    assert a and a.keys()==b.keys(),'unequal request sets'
    exact=sum(a[k]['output_ids']==b[k]['output_ids'] for k in a)
    finish_exact=sum(a[k]['finish_reason']==b[k]['finish_reason'] for k in a)
    for key in a:
        assert a[key]['prompt_ids']==b[key]['prompt_ids'],'unequal tokenized prompt'
    tokens_a=sum(x['output_tokens'] for x in a.values());tokens_b=sum(x['output_tokens'] for x in b.values())
    seconds_a=sum(x['wall_seconds'] for x in a.values());seconds_b=sum(x['wall_seconds'] for x in b.values())
    mismatches=[{'group_id':k,'ar_tokens':len(a[k]['output_ids']),'method_tokens':len(b[k]['output_ids']),
                 'first_different_position':next((i for i,(x,y) in enumerate(zip(a[k]['output_ids'],b[k]['output_ids'])) if x!=y),min(len(a[k]['output_ids']),len(b[k]['output_ids'])))}
                for k in a if a[k]['output_ids']!=b[k]['output_ids']]
    return {'count':len(a),'exact_matches':exact,'finish_matches':finish_exact,'ar_tps':tokens_a/seconds_a,'method_tps':tokens_b/seconds_b,
            'throughput_ratio':(tokens_b/seconds_b)/(tokens_a/seconds_a),'ar_output_tokens':tokens_a,
            'method_output_tokens':tokens_b,'mismatches':mismatches,
            'request_batch_size':batch_size,'ratio_note':('aggregate output tokens divided by summed batch wall times; not individual latency' if batch_size>1 else 'aggregate output tokens divided by summed request wall times; workers are independent sequential streams')}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ar',type=Path,nargs='+',required=True)
    parser.add_argument('--method',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--require-exact',action='store_true')
    parser.add_argument('--expected-count',type=int,required=True)
    args=parser.parse_args()
    result=compare(args.ar,args.method)
    assert result['count']==args.expected_count,result['count']
    write(args.out,result)
    print(json.dumps(result),flush=True)
    if args.require_exact:
        assert result['exact_matches']==result['count'] and result['finish_matches']==result['count'],result
