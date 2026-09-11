"""Extend the archived16k calibration pool without changing any old split."""
import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from .pilot_data import rows,sha,write


def groupkey(text):
    if not isinstance(text,str) or not text.strip():return None
    text=' '.join(unicodedata.normalize('NFKC',text).lower().split())
    text=re.sub(r'\d+(?:\.\d+)?','#',text)
    text=re.sub(r'[^\w#]+',' ',text)
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def main(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    assert sha(args.parquet)=='25c0fa2ca96dd8078470ad1bbc439d0d211fd8b7d22e04abf7f61e9be9bf9ec3'
    old={split:rows(args.original/f'{split}.jsonl') for split in ['train','dev','eval']}
    assert len(old['train'])==16384
    excluded={r['group_id'] for part in old.values() for r in part}
    assert len(excluded)==sum(map(len,old.values()))
    assert all(groupkey(r['problem'])==r['group_id'] for part in old.values() for r in part)
    raw=pq.read_table(args.parquet,columns=['problem','source']).to_pylist()
    seen=set();strata=collections.defaultdict(list)
    for index,row in enumerate(raw):
        key=groupkey(row['problem'])
        if key is None or key in seen:continue
        seen.add(key)
        if key not in excluded:
            strata[row['source']].append(dict(row,row_id=index,group_id=key))
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    needed=args.records-len(old['train']);assert needed>0
    total=sum(map(len,strata.values()))
    proportional={name:needed*len(items)/total for name,items in strata.items()}
    quotas={name:int(value) for name,value in proportional.items()}
    for name in sorted(proportional,key=lambda k:proportional[k]-quotas[k],reverse=True)[:needed-sum(quotas.values())]:quotas[name]+=1
    added=[];skipped=collections.Counter()
    for name,items in sorted(strata.items()):
        if quotas[name]==0:continue
        items.sort(key=lambda r:hashlib.sha256(('42:'+r['group_id']).encode()).digest())
        count=0
        for row in items:
            content=row['problem']+'\nSolve the problem and put your final answer within \\boxed{}.'
            prompt=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            if len(ids)>1024:
                skipped['prompt_over_1024']+=1;continue
            added.append(dict(row,prompt=prompt,prompt_token_ids=ids));count+=1
            if count==quotas[name]:break
        assert count==quotas[name],name
    added.sort(key=lambda r:hashlib.sha256(('split42:'+r['group_id']).encode()).digest())
    training=old['train']+added
    assert len(training)==args.records and len({r['group_id'] for r in training})==args.records
    assert not ({r['group_id'] for r in added}&excluded)
    # Verify the tokenizer/template implementation against the archived contract.
    for row in old['train'][:16]:
        content=row['problem']+'\nSolve the problem and put your final answer within \\boxed{}.'
        text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert text==row['prompt'] and tokenizer.encode(text,add_special_tokens=False)==row['prompt_token_ids']
    args.out.mkdir(parents=True,exist_ok=True)
    dest=args.out/f'train-{args.records}.jsonl'
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in training)
    if dest.exists():assert dest.read_text()==content
    else:dest.write_text(content)
    write(args.out/'provenance.json',{'source_revision':'9d8d210c9f6a36c8f3cd84045668c9b7800ef517',
          'source_sha256':sha(args.parquet),'original_train_sha256':sha(args.original/'train.jsonl'),
          'output_sha256':sha(dest),'records':len(training),'unchanged_prefix_records':len(old['train']),
          'excluded_original_groups':len(excluded),'added_source_counts':dict(collections.Counter(r['source'] for r in added)),
          'skipped':dict(skipped),'selection':'same within-source hash ordering; proportional quotas over remaining groups; stable shuffle of extension only',
          'deduplication':'original NFKC/number/punctuation template grouping; no semantic equivalence claim',
          'reserved_dev_eval_unchanged':True,'teacher_solutions_used':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--parquet',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--records',type=int,default=32768)
    main(parser.parse_args())
