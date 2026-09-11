"""Freeze tuning prompts independently of training and the final development set."""
import argparse,json
from pathlib import Path
from experiments.auf_vllm.pilot_data import rows,write,sha

def select(source,train,final,count=32):
    selected=source[128:128+count+4]
    assert len(selected)==count+4
    keys=[r['group_id'] for r in selected]
    assert len(set(keys))==len(keys)
    excluded={r['group_id'] for r in train+final}
    assert not set(keys)&excluded,'Tuning/train/final overlap'
    text_excluded={r['problem'].strip() for r in train+final if 'problem' in r}
    assert not {r['problem'].strip() for r in selected}&text_excluded,'Duplicate text across splits'
    return selected[:count],selected[count:]

def main(a):
    from transformers import AutoTokenizer
    source,train,final=rows(a.source),rows(a.train),rows(a.final)
    evaluation,warmup=select(source,train,final)
    tokenizer=AutoTokenizer.from_pretrained(a.target,local_files_only=True)
    for name,items in [('eval',evaluation),('warmup',warmup)]:
        converted=[]
        for row in items:
            extra={'enable_thinking':False} if a.family=='q8' else {}
            content=row['problem']+'\nSolve the problem and put your final answer within \\boxed{}.'
            ids=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,**extra)
            assert len(ids)+2048+16<=5120
            converted.append(dict(row,prompt_token_ids=ids))
        dest=a.out/f'{name}.json'
        if dest.exists():assert json.loads(dest.read_text())==converted
        else:write(dest,converted)
    write(a.out/'provenance.json',{'purpose':'hyperparameter selection only','count':32,'warmup_count':4,
          'source_indices':[128,164],'source_sha256':sha(a.source),'training_sha256':sha(a.train),
          'final_sha256':sha(a.final),'target':str(a.target),'family':a.family,
          'exposure':'previous fitting-validation pool, not untouched confirmation'})
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ['source','train','final','target','out']:p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--family',choices=['q8','llama','cross'],required=True);main(p.parse_args())
