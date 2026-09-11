"""Compare complete token sequences and aggregate paired decoding evidence."""
import argparse
import json
from pathlib import Path
from .pilot_data import write


def compare(ar, method):
    def read(paths):
        result={}
        caps=set()
        for path in paths:
            assert path.with_suffix('.summary.json').exists(),path
            summary=json.loads(path.with_suffix('.summary.json').read_text())
            assert summary['timing_valid']
            caps.add(summary.get('contract',summary.get('args',{}))['cap'])
            for line in path.read_text().splitlines():
                row=json.loads(line)
                assert row['group_id'] not in result,'duplicate request'
                assert row['timing_valid']
                result[row['group_id']]=row
        assert len(caps)==1
        return result,caps
    (a,caps_a),(b,caps_b)=read(ar),read(method)
    assert caps_a==caps_b,'unequal output caps'
    assert a and a.keys()==b.keys(),'unequal request sets'
    exact=sum(a[k]['output_ids']==b[k]['output_ids'] for k in a)
    for key in a:
        assert a[key]['prompt_ids']==b[key]['prompt_ids'],'unequal tokenized prompt'
    tokens_a=sum(x['output_tokens'] for x in a.values());tokens_b=sum(x['output_tokens'] for x in b.values())
    seconds_a=sum(x['wall_seconds'] for x in a.values());seconds_b=sum(x['wall_seconds'] for x in b.values())
    mismatches=[{'group_id':k,'ar_tokens':len(a[k]['output_ids']),'method_tokens':len(b[k]['output_ids']),
                 'first_different_position':next((i for i,(x,y) in enumerate(zip(a[k]['output_ids'],b[k]['output_ids'])) if x!=y),min(len(a[k]['output_ids']),len(b[k]['output_ids'])))}
                for k in a if a[k]['output_ids']!=b[k]['output_ids']]
    return {'count':len(a),'exact_matches':exact,'ar_tps':tokens_a/seconds_a,'method_tps':tokens_b/seconds_b,
            'throughput_ratio':(tokens_b/seconds_b)/(tokens_a/seconds_a),'ar_output_tokens':tokens_a,
            'method_output_tokens':tokens_b,'mismatches':mismatches,
            'ratio_note':'aggregate output tokens divided by summed request wall times; workers are independent sequential streams'}


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
        assert result['exact_matches']==result['count'],result
