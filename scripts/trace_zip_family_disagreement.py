"""Observe first committed disagreement; replay its exact cache and inputs."""
import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from relayspec.dflash import import_official_dflash
from relayspec.benchmarking import benchmark_turns
from relayspec.generation import native_autoregressive_generate, relay_dflash_generate, cross_family_relay_dflash_generate, accepted_block_length
from relayspec.relay import TargetFeatureRelay
from relayspec.vocab_bridge import load_vocab_intersection


def cache_copy(cache, length):
    c = copy.deepcopy(cache)
    c.crop(length)
    return c


@torch.inference_mode()
def main():
    rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    pair = 'llama' if rank < 2 else 'cross'
    ids = ['math500:test/number_theory/931.json','math500:test/number_theory/521.json'] if pair == 'llama' else ['math500:test/number_theory/521.json','math500:test/intermediate_algebra/207.json']
    pid = ids[rank % 2]
    root = Path(os.environ['TRACE_OUTPUT']) / f'{pair}-{rank}'
    root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(f'reports/zip-family-pilot-20260907/{pair}-epoch20.yaml').read_text())
    cache_dir = os.environ['TRANSFORMERS_CACHE']
    def load(spec):
        return AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], cache_dir=cache_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    source = load(cfg['source_trunk']['model'])
    embedding, head = source.model.embed_tokens, source.lm_head
    del source
    target = load(cfg['target'])
    dc, _ = import_official_dflash(os.environ['DFLASH_SOURCE'],cfg['proposer']['source_commit'])
    draft = dc.from_pretrained(cfg['proposer']['id'],revision=cfg['proposer']['revision'],cache_dir=cache_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    tok = AutoTokenizer.from_pretrained(cfg['target']['id'],revision=cfg['target']['revision'],cache_dir=cache_dir,local_files_only=True)
    st = AutoTokenizer.from_pretrained(cfg['source_trunk']['model']['id'],revision=cfg['source_trunk']['model']['revision'],cache_dir=cache_dir,local_files_only=True)
    q = torch.load(cfg['relay_probe']['checkpoint_path'],weights_only=True,map_location='cpu')
    taps = q['target_layer_ids']
    relay = TargetFeatureRelay(target_hidden_size=target.config.hidden_size,num_taps=len(taps),draft_hidden_size=draft.config.hidden_size,eps=target.config.rms_norm_eps,normalize_input=False)
    relay.load_state_dict(q['relay']);relay=relay.cuda().bfloat16().eval()
    records=json.loads(Path('configs/eval_manifest.json').read_text())['records']
    record=next(r for r in records if r['problem_id']==pid)
    rendered_prompt=benchmark_turns(record)[0]
    encoded=tok.apply_chat_template([{'role':'user','content':rendered_prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt')
    inputs=(encoded['input_ids'] if hasattr(encoded,'keys') else encoded).cuda()
    original=target.forward
    ar_logits={};ar_hidden={};ar_cache=None;ar_history=[];spec_history=[];found=None
    mode='ar';last_hidden=None;ar_ids=None
    def head_hook(module,args):
        nonlocal last_hidden
        last_hidden=args[0].detach()
    handle=target.lm_head.register_forward_pre_hook(head_hook)
    def traced(input_ids,**kwargs):
        nonlocal ar_cache,ar_history,spec_history,found
        cache=kwargs.get('past_key_values')
        before=int(cache.get_seq_length()) if cache is not None else 0
        fed=input_ids[0].tolist()
        out=original(input_ids,**kwargs)
        if mode=='ar':
            ar_history=ar_history[:before]+fed
            index=before+len(fed)
            ar_logits[index]=out.logits[0,-1].detach().cpu()
            ar_hidden[index]=last_hidden[0,-1].detach().cpu()
            ar_cache=out.past_key_values
        elif mode=='spec':
            actual_prefix=spec_history[:before]
            spec_history=actual_prefix+fed
            if before>0 and found is None:
                posterior=out.logits.argmax(-1)
                accepted=accepted_block_length(input_ids,posterior) if len(fed)>1 else 1
                for j in range(accepted):
                    index=before+j+1
                    if index>=len(ar_ids):break
                    prefix=actual_prefix+fed[:j+1]
                    if prefix!=ar_ids[:index]:break
                    if int(posterior[0,j])!=ar_ids[index]:
                        found={'index':index,'j':j,'before':before,'accepted':accepted,'block':input_ids.clone(),'cache':cache_copy(out.past_key_values,before),'spec_logits':out.logits[0,j].detach().cpu(),'spec_hidden':last_hidden[0,j].detach().cpu(),'prefix':prefix,'ar_token':ar_ids[index],'spec_token':int(posterior[0,j])}
                        print(json.dumps({'event':'first_divergence','pair':pair,'rank':rank,'token':index-inputs.shape[1],'cache_length':before,'block_length':len(fed),'within_block':j}),flush=True)
                        break
        return out
    target.forward=traced
    ar=native_autoregressive_generate(target,input_ids=inputs,max_new_tokens=512,stop_token_ids=[tok.eos_token_id],temperature=0.,return_stats=True)
    ar_ids=ar.output_ids[0].tolist()
    mode='spec'
    common=dict(relay=relay,relay_target_layer_ids=taps,native_target=target,source_embedding=embedding,source_lm_head=head,input_ids=inputs,max_new_tokens=512,stop_token_ids=[tok.eos_token_id],temperature=0.,block_size=cfg['benchmark']['block_size'],return_stats=True)
    if pair=='cross':
        b=cfg['benchmark'];s2t=load_vocab_intersection(Path(b['vocab_bridge_dir'])/b['vocab_bridge_source_to_target']);t2s=load_vocab_intersection(Path(b['vocab_bridge_dir'])/b['vocab_bridge_target_to_source'])
        spec=cross_family_relay_dflash_generate(draft,source_tokenizer=st,target_tokenizer=tok,source_to_target_intersection=s2t,target_to_source_intersection=t2s,**common)
    else:spec=relay_dflash_generate(draft,**common)
    target.forward=original;handle.remove()
    spec_ids=spec.output_ids[0].tolist()
    first=next((i for i,(a,b) in enumerate(zip(ar_ids,spec_ids)) if a!=b),None)
    result={'pair':pair,'problem_id':pid,'rank':rank,'job_id':os.environ['SLURM_JOB_ID'],'ar_ids':ar_ids,'spec_ids':spec_ids,'first_divergence_generated_index':None if first is None else first-inputs.shape[1],'exact_match':ar_ids==spec_ids}
    previous=Path('outputs/28857')/(pair+'-epoch20')
    prior=[json.loads(line) for p in previous.glob('benchmark-rank*.jsonl') for line in p.read_text().splitlines()]
    def output_hash(ids):
        return hashlib.sha256(torch.tensor(ids[inputs.shape[1]:],dtype=torch.int32).numpy().tobytes()).hexdigest()
    result['matches_original_run']={}
    for method,token_ids in [('native_ar',ar_ids),('relay_p' if pair=='llama' else 'relay_p_cross_family',spec_ids)]:
        previous_row=next(r for r in prior if r['problem_id']==pid and r['method']==method)
        result['matches_original_run'][method]=output_hash(token_ids)==previous_row['output_hash']
    result['rendered_user_prompt']=rendered_prompt
    if found is not None:
        assert first==found['index'],(first,found['index'])
        i,j,before=found['index'],found['j'],found['before'];block=found['block']
        a,b=found['ar_token'],found['spec_token']
        def describe(logits):
            l=logits.float().cpu();top=l.topk(5)
            return {'argmax':int(l.argmax()),'ar_logit':float(l[a]),'spec_logit':float(l[b]),'ar_minus_spec':float(l[a]-l[b]),'top_ids':top.indices.tolist(),'top_logits':top.values.tolist(),'maximum_tie_count':int((l==l.max()).sum())}
        result.update({'cache_length':before,'block_length':block.shape[1],'position_within_block':j,'prefix_token_ids_equal':found['prefix']==ar_ids[:i],'ar_token':a,'spec_token':b,'ar_token_text':tok.decode([a]),'spec_token_text':tok.decode([b]),'original_ar':describe(ar_logits[i]),'original_spec':describe(found['spec_logits'])})
        def replay(cache,sequence,sequential=False):
            if sequential:
                for pos in range(j+1):out=original(sequence[:,pos:pos+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
                return out.logits[0,-1]
            out=original(sequence,past_key_values=cache,use_cache=True)
            return out.logits[0,j]
        result['same_spec_cache_full_block']=describe(replay(cache_copy(found['cache'],before),block))
        result['same_spec_cache_truncated_block']=describe(replay(cache_copy(found['cache'],before),block[:,:j+1]))
        result['same_spec_cache_single_token_steps']=describe(replay(cache_copy(found['cache'],before),block,True))
        result['ar_cache_full_block']=describe(replay(cache_copy(ar_cache,before),block))
        result['ar_cache_single_token_steps']=describe(replay(cache_copy(ar_cache,before),block,True))
        # Recompute the output projection only, preserving the observed hidden states.
        weight=target.lm_head.weight.float()
        for name,h in [('ar',ar_hidden[i]),('spec',found['spec_hidden'])]:
            logits=torch.nn.functional.linear(h.cuda().float(),weight)
            result[name+'_hidden_fp32_head']=describe(logits)
        result['hidden_max_abs_difference']=float((ar_hidden[i].float()-found['spec_hidden'].float()).abs().max())
        del weight
        # Freeze a small reproducer. Prefix IDs + exact block are enough to regenerate
        # both caches; preserve vectors to audit precision effects without a GPU.
        torch.save({'block_ids':block.cpu(),'prefix_ids':found['prefix'],'ar_hidden':ar_hidden[i],'spec_hidden':found['spec_hidden'],'ar_logits':ar_logits[i],'spec_logits':found['spec_logits']},root/'divergence.pt')
    (root/'diagnosis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('ar_ids','spec_ids')}),flush=True)

if __name__=='__main__':main()
