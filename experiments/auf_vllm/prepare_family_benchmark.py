"""Freeze 128 Numina development prompts, separate from fitting validation."""
import argparse
import json
from pathlib import Path
from .pilot_data import rows,write,sha


def main(args):
    from transformers import AutoTokenizer
    target=args.target_path or args.models/{'llama':'llama3-target','q14':'qwen14-target','q8':'qwen8-target'}[args.family]
    tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True)
    source=rows(args.manifest)
    evaluation,warmup=source[:128],source[-4:]
    assert len({r['group_id'] for r in evaluation})==128
    assert not ({r['group_id'] for r in evaluation}&{r['group_id'] for r in warmup})
    for name,items in [('eval',evaluation),('warmup',warmup)]:
        converted=[]
        for row in items:
            content=row['problem']+'\nSolve the problem and put your final answer within \\boxed{}.'
            extra={} if args.family in ('llama','cross') else {'enable_thinking':False}
            text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,**extra)
            ids=tokenizer.encode(text,add_special_tokens=False)
            assert len(ids)+2048+16<=5120
            converted.append(dict(row,prompt_token_ids=ids))
        path=args.out/f'{name}.json'
        if path.exists():
            assert json.loads(path.read_text())==converted
        else:
            write(path,converted)
    write(args.out/'provenance.json',{'source':str(args.manifest),'sha256':sha(args.manifest),
          'family':args.family,'split':'Numina development; not untouched confirmation or MATH500',
          'count':128,'warmup_count':4,'target_adapters':None})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=['llama','q14','q8','cross'],required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--target-path',type=Path)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
