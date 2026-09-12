"""Warmed same-family vLLM decoding with per-request telemetry and token IDs."""
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from .pilot_data import write,sha


def serial(value):
    if dataclasses.is_dataclass(value):return dataclasses.asdict(value)
    if hasattr(value,'__dict__'):return vars(value)
    return str(value)


def main(args):
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    assert os.environ.get('VLLM_BATCH_INVARIANT')=='1'
    assert 0<=args.worker_index<args.workers
    batch_size=getattr(args,'request_batch_size',1)
    assert 1<=batch_size<=128
    if batch_size>1:assert args.workers==1, 'Fixed-batch serving is a single-GPU workload'
    target=args.target_path or args.models/{'llama':'llama3-target','q14':'qwen14-target','q8':'qwen8-target'}[args.family]
    assert not (args.export and args.native_draft)
    if args.family=='cross':
        assert args.target_path is not None,'Cross target must be explicit'
    if args.export:
        os.environ['TRANSFER_MAPPED']='1'
        runtime=str(Path(__file__).resolve().parent/'runtime_zip')
        sys.path.insert(0,runtime)
        os.environ['PYTHONPATH']=runtime+os.pathsep+os.environ.get('PYTHONPATH','')
        from mapper_runtime import install
        install()
        if args.family=='cross':
            os.environ['CROSS_BRIDGE']='1'
            os.environ['CROSS_TARGET_TOKENIZER']=str(target)
            assert os.environ.get('CROSS_SOURCE_TOKENIZER'),'Cross source tokenizer must be explicit'
            from experiments.handoff_transfer.cross.runtime import install as install_cross
            install_cross()
    config=dict(model=str(target),dtype='bfloat16',max_model_len=5120,max_num_seqs=max(8,batch_size),
                max_num_batched_tokens=8192,gpu_memory_utilization=.9 if args.family=='q14' else .8,
                enable_prefix_caching=False,generation_config='vllm',async_scheduling=False,
                seed=0,disable_log_stats=False,worker_extension_cls='experiments.auf_vllm.family_metrics.FamilyMetricsWorker',
                compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'})
    if args.export:
        export_config=json.loads((args.export/'config.json').read_text())
        block=export_config['block_size']
        config['speculative_config']={'method':'dflash','model':str(args.export),'num_speculative_tokens':block-1}
    elif args.native_draft:
        assert args.family=='q8' and args.mode=='native'
        block=json.loads((args.native_draft/'config.json').read_text())['block_size']
        config['speculative_config']={'method':'dflash','model':str(args.native_draft),'num_speculative_tokens':block-1}
    else:
        assert args.mode=='ar'
    if args.profile_dir:
        config['profiler_config']={'profiler':'torch','torch_profiler_dir':str(args.profile_dir.resolve()),
             'torch_profiler_with_stack':False,'torch_profiler_record_shapes':True,'torch_profiler_with_memory':True,
             'torch_profiler_with_flops':True,'max_iterations':64,'ignore_frontend':True,'detailed_trace_annotation':True}
    tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True)
    params=SamplingParams(temperature=0,top_p=1.,top_k=-1,min_p=0.,repetition_penalty=1.,
                          presence_penalty=0.,frequency_penalty=0.,seed=0,max_tokens=args.cap,
                          stop_token_ids=[tokenizer.eos_token_id],ignore_eos=False)
    rows=json.loads((args.data/'eval.json').read_text())[:args.count]
    assert len(rows)==args.count and len({r['group_id'] for r in rows})==args.count
    selected=list(enumerate(rows))[args.worker_index::args.workers]
    warm=json.loads((args.data/'warmup.json').read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    suffix=f'-b{batch_size}' if batch_size>1 else ''
    path=args.out/f'{args.mode}-r{args.repeat}-w{args.worker_index}{suffix}.jsonl'
    summary_path=path.with_suffix('.summary.json')
    contract={'family':args.family,'mode':args.mode,'manifest_sha256':sha(args.data/'eval.json'),
              'count':args.count,'cap':args.cap,'worker_index':args.worker_index,'workers':args.workers,
              'repeat':args.repeat,'export_sha256':sha(args.export/'model.safetensors') if args.export else None,
              'runtime_config':config}
    if batch_size>1:contract.update(request_batch_size=batch_size,workload='fixed synchronous request batches')
    if summary_path.exists():
        assert json.loads(summary_path.read_text())['contract']==contract
        return
    if path.exists():
        path.rename(path.with_name(path.name+f'.incomplete-{time.time_ns()}'))
    start=time.perf_counter();llm=LLM(**config);setup=time.perf_counter()-start
    llm.collective_rpc('sd_install_monitor')
    assets=llm.collective_rpc('sd_verify_family_assets',args=(str(target),str(args.export) if args.export else None,
                                str(args.native_draft) if args.native_draft else None))
    extra={} if args.family in ('llama','cross') else {'enable_thinking':False}
    text=tokenizer.apply_chat_template([{'role':'user','content':'What is 2 + 2? Answer with only the number.'}],tokenize=False,add_generation_prompt=True,**extra)
    reference=llm.generate([{'prompt_token_ids':tokenizer.encode(text,add_special_tokens=False)}],SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)[0].outputs[0]
    assert reference.text.strip()=='4',reference.text
    start=time.perf_counter()
    for start_index in range(0,len(warm),batch_size):
        llm.generate([{'prompt_token_ids':row['prompt_token_ids']} for row in warm[start_index:start_index+batch_size]],params,use_tqdm=False)
    warm_seconds=time.perf_counter()-start
    before=llm.collective_rpc('sd_stats')
    if args.profile_dir:llm.start_profile()

    def counters():
        return {serial(x)['name']:serial(x).get('value',0) for x in llm.get_metrics()}

    batch_measurements=None
    if batch_size>1:
        from .batch_measurement import measure
        batch_measurements=measure(llm,selected,params,path,batch_size,counters,bool(args.profile_dir))
    else:
        with path.open('x') as handle:
            for index,row in selected:
                status0=llm.collective_rpc('sd_stats')[0];counts0=counters()
                start=time.perf_counter()
                output=llm.generate([{'prompt_token_ids':row['prompt_token_ids']}],params,use_tqdm=False)[0]
                seconds=time.perf_counter()-start
                status1=llm.collective_rpc('sd_stats')[0];cold=None
                if any(status0[k]!=status1[k] for k in ['jit_events','teacher_graph_captures']):
                    cold=seconds;counts0=counters();start=time.perf_counter()
                    output=llm.generate([{'prompt_token_ids':row['prompt_token_ids']}],params,use_tqdm=False)[0]
                    seconds=time.perf_counter()-start
                    status2=llm.collective_rpc('sd_stats')[0]
                    assert all(status1[k]==status2[k] for k in ['jit_events','teacher_graph_captures'])
                counts1=counters()
                delta=lambda key:counts1.get('vllm:'+key,0)-counts0.get('vllm:'+key,0)
                result=output.outputs[0]
                record={'index':index,'group_id':row['group_id'],'row_id':row['row_id'],'output_ids':list(result.token_ids),
                        'output_text':result.text,'output_tokens':len(result.token_ids),'prompt_ids':row['prompt_token_ids'],
                        'prompt_tokens':len(row['prompt_token_ids']),'wall_seconds':seconds,'cold_wall_seconds':cold,
                        'finish_reason':result.finish_reason,'verification_iterations':delta('spec_decode_num_drafts'),
                        'accepted_draft_tokens':delta('spec_decode_num_accepted_tokens'),'timing_valid':not bool(args.profile_dir)}
                handle.write(json.dumps(record,default=serial)+'\n');handle.flush()
                if (index//args.workers+1)%16==0:
                    print(json.dumps({'mode':args.mode,'completed':index//args.workers+1,'last_seconds':seconds}),flush=True)
    if args.profile_dir:llm.stop_profile()
    write(summary_path,{'contract':contract,'setup_seconds':setup,'warmup_seconds':warm_seconds,'warmup_request_batch_size':batch_size,
                        'asset_checks':assets,'gpu_before':before,'gpu_after':llm.collective_rpc('sd_stats'),
                        'job_id':os.environ.get('SLURM_JOB_ID'),'timing_valid':not bool(args.profile_dir),
                        'batch_measurements':batch_measurements,'timing_contract':('fixed-batch generate wall; per-row wall is amortized, not individual latency' if batch_size>1 else 'per-request generate wall; compilation-affected request retried once, cold time retained; prefix cache off')})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=['llama','q14','q8','cross'],required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--target-path',type=Path)
    parser.add_argument('--native-draft',type=Path)
    parser.add_argument('--export',type=Path)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['matrix','ar','native','zip','ce','auf','draft-ce','draft-auf','fusion-ce','fusion-auf','handoff-r56','handoff-five','normal','handoff-ce','handoff-capacity','handoff-draft'],required=True)
    parser.add_argument('--count',type=int,default=128)
    parser.add_argument('--cap',type=int,default=2048)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--worker-index',type=int,default=0)
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--profile-dir',type=Path)
    parser.add_argument('--request-batch-size',type=int,choices=[1,4,8,16,32,64,128],default=1)
    main(parser.parse_args())
