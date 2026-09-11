"""Generate only the newly added half of the frozen32768-record manifest."""
import argparse
import json
import os
import time
from pathlib import Path
from .pilot_data import rows,write,sha


def main(args):
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    assert os.environ.get('VLLM_BATCH_INVARIANT')=='1'
    assert 0<=args.shard<64
    all_rows=rows(args.manifest)
    assert len(all_rows)==32768
    start=16384+256*args.shard
    selected=all_rows[start:start+256]
    assert len({r['group_id'] for r in selected})==256
    out=args.out/f'part-{args.shard:05d}'
    contract={'manifest_sha256':sha(args.manifest),'offset':start,'records':256,
              'target':str(args.target),'target_revision':'b968826d9c46dd6066d109eabc6255188de91218',
              'target_adapters':None,'cap':4096,'temperature':0,'seed':42,
              'max_num_seqs':64,'max_num_batched_tokens':8192,'batch_invariant':True}
    if (out/'generation.json').exists():
        report=json.loads((out/'generation.json').read_text())
        assert report['contract']==contract and report['sha256']==sha(out/'train.json')
        return
    assert not (out/'train.json').exists(),'Inspect incomplete shard before replacing labels'
    tokenizer=AutoTokenizer.from_pretrained(args.target,local_files_only=True)
    prompts=[]
    for row in selected:
        content=row['problem']+'\nSolve the problem and put your final answer within \\boxed{}.'
        text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(text,add_special_tokens=False)
        assert ids==row['prompt_token_ids'] and len(ids)<=1024
        prompts.append(ids)
    llm=LLM(model=str(args.target),dtype='bfloat16',max_model_len=5120,max_num_seqs=64,
            max_num_batched_tokens=8192,gpu_memory_utilization=.8,enable_prefix_caching=False,
            generation_config='vllm',seed=42,
            compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'})
    params=SamplingParams(temperature=0,max_tokens=4096,seed=42,stop_token_ids=[tokenizer.eos_token_id])
    start_time=time.perf_counter()
    outputs=llm.generate([{'prompt_token_ids':ids} for ids in prompts],params,use_tqdm=False)
    generated=[]
    for row,prompt,response in zip(selected,prompts,outputs):
        result=response.outputs[0];tokens=list(result.token_ids)
        generated.append({'row_id':row['row_id'],'group_id':row['group_id'],'prompt_token_ids':prompt,
                          'output_ids':tokens,'full_ids':prompt+tokens,'temperature':0,'thinking':False,
                          'finish_reason':result.finish_reason})
    assert len(generated)==256
    write(out/'train.json',generated);write(out/'dev.json',[])
    write(out/'generation.json',{'contract':contract,'sha256':sha(out/'train.json'),
          'seconds':time.perf_counter()-start_time,'tokens':sum(len(r['output_ids']) for r in generated),
          'job_id':os.environ.get('SLURM_JOB_ID'),'status':'complete'})
    print(json.dumps({'shard':args.shard,'records':256,'seconds':time.perf_counter()-start_time}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True)
    main(p.parse_args())
