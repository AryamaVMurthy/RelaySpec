"""Standalone Transformers replication; no vLLM import or engine."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config
from relayspec.dflash import import_official_dflash
from relayspec.generation import native_autoregressive_generate, relay_dflash_generate
from experiments.handoff_transfer.pack_transfer import sha, write


@torch.inference_mode()
def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out/f'{a.mode}-r0-w0.jsonl'
    assert not path.exists(), 'Use a fresh output directory; retain prior runs'
    tokenizer=AutoTokenizer.from_pretrained(a.target,local_files_only=True)
    eos=[tokenizer.eos_token_id]
    target=AutoModelForCausalLM.from_pretrained(a.target,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    runtime={'backend':'transformers','target':str(a.target),'dtype':'bfloat16',
             'attention':'sdpa','selective_capture':True,'temperature':0,'stop_ids':eos}
    contract={'family':a.family,'mode':a.mode,'count':a.count,'cap':a.cap,'repeat':0,
              'manifest_sha256':sha(a.data/'eval.json'),'runtime_config':runtime,
              'export_sha256':None if a.export is None else sha(a.export/'model.safetensors')}
    if a.mode=='ar':
        assert a.export is None
        def generate(ids):
            return native_autoregressive_generate(target,input_ids=ids,max_new_tokens=a.cap,
                stop_token_ids=eos,temperature=0,return_stats=True)
    else:
        assert a.export is not None
        config=json.loads((a.export/'config.json').read_text())
        draft_class,_=import_official_dflash(a.dflash_source,'94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756')
        cfg=Qwen3Config.from_pretrained(a.export,local_files_only=True)
        cfg._attn_implementation='sdpa'
        draft=draft_class(cfg).to(dtype=torch.bfloat16)
        weights=load_file(str(a.export/'model.safetensors'))
        fusion=weights['fc.weight']
        draft.fc=nn.Linear(fusion.shape[1],fusion.shape[0],bias=False,dtype=torch.bfloat16)
        expected=set(draft.state_dict())
        assert expected==set(weights)-{'embed_tokens.weight','lm_head.weight'}
        draft.load_state_dict({k:weights[k] for k in expected},strict=True)
        assert all(torch.equal(v,weights[k]) for k,v in draft.state_dict().items())
        draft=draft.cuda().eval().requires_grad_(False)
        embed=nn.Embedding.from_pretrained(weights['embed_tokens.weight'],freeze=True).cuda()
        head=nn.Linear(weights['lm_head.weight'].shape[1],weights['lm_head.weight'].shape[0],bias=False,dtype=torch.bfloat16)
        head.weight=nn.Parameter(weights['lm_head.weight'],requires_grad=False);head=head.cuda()
        relay=draft.fc
        if a.mode=='normal':
            eps=config['relayspec_normal_input_eps']
            relay=nn.Sequential(nn.RMSNorm(fusion.shape[1],eps=eps,elementwise_affine=False),draft.fc).cuda()
        taps=tuple(config['dflash_config']['target_layer_ids'])
        def generate(ids):
            return relay_dflash_generate(draft,relay=relay,relay_target_layer_ids=taps,
                native_target=target,source_embedding=embed,source_lm_head=head,input_ids=ids,
                max_new_tokens=a.cap,stop_token_ids=eos,temperature=0,return_stats=True,
                selective_capture=True,release_capture_buffers=True)
        del weights
    rows=json.loads((a.data/'eval.json').read_text())[:a.count]
    assert len(rows)==a.count and len({r['group_id'] for r in rows})==a.count
    for row in json.loads((a.data/'warmup.json').read_text()):
        generate(torch.tensor([row['prompt_token_ids']],device='cuda'))
    torch.cuda.synchronize()
    with path.open('x') as stream:
        for index,row in enumerate(rows):
            ids=torch.tensor([row['prompt_token_ids']],device='cuda')
            torch.cuda.synchronize();start=time.perf_counter();result=generate(ids)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            output=result.output_ids[0,ids.shape[1]:].tolist()
            assert 0<len(output)<=a.cap
            record={'group_id':row['group_id'],'prompt_ids':row['prompt_token_ids'],'output_ids':output,
                    'output_tokens':len(output),'wall_seconds':seconds,'timing_valid':True,
                    'finish_reason':'stop' if output[-1] in eos else 'length',
                    'target_calls':result.target_calls,'draft_calls':result.draft_calls}
            stream.write(json.dumps(record)+'\n');stream.flush()
            if (index+1)%16==0:print(json.dumps({'mode':a.mode,'completed':index+1}),flush=True)
    write(path.with_suffix('.summary.json'),{'contract':contract,'timing_valid':True,
          'timing_contract':'Synchronized request wall time including prefill; warmed standalone Transformers',
          'torch_version':torch.__version__,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('target','data','out','dflash-source'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--export',type=Path)
    p.add_argument('--family',choices=['q8','llama'],required=True)
    p.add_argument('--mode',choices=['ar','normal','zip','handoff-r56','handoff-five'],required=True)
    p.add_argument('--count',type=int,default=128);p.add_argument('--cap',type=int,default=2048)
    main(p.parse_args())
