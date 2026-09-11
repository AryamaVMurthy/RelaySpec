"""Pin development/confirmation requests after a recorded local exposure audit."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from .extend_data_manifest import groupkey
from .pilot_data import sha, write, rows


TEXT_KEYS={'problem','question','question_content','prompt','instruction','input_text'}


def exposed_texts(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in TEXT_KEYS and isinstance(item,str):
                yield item
            elif key=='turns' and isinstance(item,list):
                yield from (x for x in item if isinstance(x,str))
            elif key in {'messages','conversations'} and isinstance(item,list):
                for message in item:
                    if isinstance(message,dict) and (message.get('role')=='user' or message.get('from')=='human'):
                        text=message.get('content',message.get('value'))
                        if isinstance(text,str):yield text
            if isinstance(item,(dict,list)):
                yield from exposed_texts(item)
    elif isinstance(value,list):
        for item in value:
            if isinstance(item,(dict,list)):yield from exposed_texts(item)


def audit(paths):
    groups=set();inventory=[];errors=[];seen=set()
    for index,path in enumerate(sorted(paths)):
        actual=path.resolve()
        if actual in seen:continue
        seen.add(actual)
        before=len(groups)
        try:
            if path.suffix=='.jsonl':
                with path.open() as f:
                    for line in f:
                        if line.strip():groups.update(filter(None,map(groupkey,exposed_texts(json.loads(line)))))
            else:
                groups.update(filter(None,map(groupkey,exposed_texts(json.loads(path.read_text())))))
        except (ValueError,UnicodeError,OSError) as exc:
            errors.append({'path':str(path),'error':str(exc)})
        inventory.append({'path':str(path),'sha256':sha(path),'new_groups':len(groups)-before})
        if index%500==0:print(json.dumps({'audited_files':index+1,'groups':len(groups)}),flush=True)
    return groups,inventory,errors


def candidates(args):
    import pyarrow.parquet as pq
    result={};sources=[]
    for workload,path,field in [('math',args.math,'problem'),('gsm',args.gsm,'question')]:
        data=pq.read_table(path).to_pylist()
        result[workload]=[{'row_id':f'{workload}:{i}','problem':r[field],
                           'source_file':str(path),'source_index':i,
                           'reference':r.get('solution',r.get('answer')),
                           'metadata':{k:v for k,v in r.items() if k not in {field,'solution','answer'}}}
                          for i,r in enumerate(data)]
        sources.append({'path':str(path),'sha256':sha(path),'records':len(data)})
    code=[]
    for path in sorted(args.code.glob('test*.jsonl')):
        data=rows(path)
        for i,r in enumerate(data):
            problem=r['question_content']
            starter=r.get('starter_code','')
            if starter:problem+='\n\nStarter code:\n'+starter
            code.append({'row_id':f"lcb:{r['platform']}:{r['question_id']}",'problem':problem,
                         'question_content':r['question_content'],
                         'source_file':str(path),'source_index':i,
                         'metadata':{k:r.get(k) for k in ['platform','question_id','contest_date','difficulty']}})
        sources.append({'path':str(path),'sha256':sha(path),'records':len(data)})
    result['code']=code
    data=rows(args.chat)
    result['chat']=[{'row_id':f'sharegpt-derived:{i}','problem':r['instruction'],
                     'source_file':str(args.chat),'source_index':i,'metadata':{}}
                    for i,r in enumerate(data)]
    sources.append({'path':str(args.chat),'sha256':sha(args.chat),'records':len(data)})
    return result,sources


def content(row,workload):
    text=row['problem']
    if workload in {'math','gsm'}:
        text+='\nSolve the problem and put your final answer within \\boxed{}.'
    elif workload=='code':
        text+='\nWrite a correct Python 3 solution. Return the solution in a Python code block.'
    return text


def main(args):
    from transformers import AutoTokenizer
    assert not args.out.exists(),'Refuse to overwrite frozen benchmark manifests'
    paths=list(args.exposure_file)
    for root in args.exposure_root:
        for folder in ['configs','reports']:
            paths.extend((root/folder).rglob('*.json'))
            paths.extend((root/folder).rglob('*.jsonl'))
    excluded,inventory,errors=audit(paths)
    write(args.audit_report,{'files':inventory,'group_count':len(excluded),'errors':errors,
           'scope':'Provided local configs/reports and full calibration manifests only; no semantic or pretraining decontamination claim'})
    assert not errors, 'Inspect exposure parsing failures before declaring new confirmation requests'
    raw,sources=candidates(args)
    tokenizers={'qwen':AutoTokenizer.from_pretrained(args.qwen_tokenizer,local_files_only=True),
                'llama':AutoTokenizer.from_pretrained(args.llama_tokenizer,local_files_only=True)}
    chosen_groups=set();manifest_report={}
    for workload,items in raw.items():
        random.Random(args.seed).shuffle(items)
        selected=[];skipped=collections.Counter();seen=set()
        for row in items:
            key=groupkey(row['problem'])
            if not key or key in seen:
                skipped['duplicate_or_empty']+=1;continue
            seen.add(key)
            overlap_keys={key,groupkey(row.get('question_content',row['problem']))}
            if overlap_keys & (excluded | chosen_groups):
                skipped['exposed_or_other_workload']+=1;continue
            prompt=content(row,workload);tokens={}
            for family,tok in tokenizers.items():
                extra={'enable_thinking':False} if family=='qwen' else {}
                text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,**extra)
                tokens[family]=tok.encode(text,add_special_tokens=False)
            if any(len(ids)>3056 for ids in tokens.values()):
                skipped['prompt_above_3056']+=1;continue
            selected.append(dict(row,group_id=key,workload=workload,prompt_content=prompt,
                                 prompt_ids_by_family=tokens))
            chosen_groups.update(overlap_keys)
            if len(selected)==260:break
        assert len(selected)==260,(workload,len(selected),dict(skipped))
        # Selection is committed before any model evaluation. Confirmation is never used for tuning.
        split_rows={'development':selected[:128],'confirmation':selected[128:256],'warmup':selected[256:]}
        manifest_report[workload]={'candidate_count':len(items),'skipped_until_full':dict(skipped),'splits':{}}
        for split,data in split_rows.items():
            dest=args.out/workload/f'{split}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(''.join(json.dumps(row)+'\n' for row in data))
            manifest_report[workload]['splits'][split]={'records':len(data),'sha256':sha(dest)}
        print(json.dumps({'workload':workload,'selected':len(selected),'skipped':dict(skipped)}),flush=True)
    write(args.out/'provenance.json',{'seed':args.seed,'sources':sources,'workloads':manifest_report,
          'exposure_report_sha256':sha(args.audit_report),'exposed_groups':len(excluded),
          'scope':'MATH test; GSM8K test; pinned LiveCodeBench releases; ShareGPT-derived single-turn instructions from UltraFeedback',
          'output_cap':2048,'natural_eos':True,'max_prompt_tokens':3056,
          'confirmation_policy':'Do not evaluate until selection rules and all compared checkpoints are frozen',
          'limitation':'Normalized exact/template exclusion over inventoried local artifacts; not proof of semantic or pretraining independence'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['math','gsm','code','chat','qwen-tokenizer','llama-tokenizer','out','audit-report']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--exposure-root',type=Path,action='append',default=[])
    p.add_argument('--exposure-file',type=Path,action='append',default=[])
    p.add_argument('--seed',type=int,default=20260911)
    main(p.parse_args())
