"""Synchronous request-batch throughput; amortized wall time is not latency."""
import json,time


def measure(llm,selected,params,path,batch_size,counters,profile=False):
    assert batch_size in (4,8)
    batches=[]
    with path.open('x') as handle:
        for batch_id,start_index in enumerate(range(0,len(selected),batch_size)):
            group=selected[start_index:start_index+batch_size]
            prompts=[{'prompt_token_ids':row['prompt_token_ids']} for _,row in group]
            stats0=llm.collective_rpc('sd_stats')[0];counts0=counters();start=time.perf_counter()
            outputs=llm.generate(prompts,params,use_tqdm=False);seconds=time.perf_counter()-start
            stats1=llm.collective_rpc('sd_stats')[0];cold=None
            if any(stats0[k]!=stats1[k] for k in ('jit_events','teacher_graph_captures')):
                cold=seconds;counts0=counters();start=time.perf_counter()
                outputs=llm.generate(prompts,params,use_tqdm=False);seconds=time.perf_counter()-start
                stats2=llm.collective_rpc('sd_stats')[0]
                assert all(stats1[k]==stats2[k] for k in ('jit_events','teacher_graph_captures'))
            assert len(outputs)==len(group)
            counts1=counters()
            delta=lambda key:counts1.get('vllm:'+key,0)-counts0.get('vllm:'+key,0)
            token_count=0
            for (index,row),output in zip(group,outputs):
                assert list(output.prompt_token_ids)==row['prompt_token_ids']
                result=output.outputs[0];tokens=list(result.token_ids);token_count+=len(tokens)
                record=dict(index=index,group_id=row['group_id'],row_id=row['row_id'],
                    output_ids=tokens,output_text=result.text,output_tokens=len(tokens),
                    prompt_ids=row['prompt_token_ids'],prompt_tokens=len(row['prompt_token_ids']),
                    wall_seconds=seconds/len(group),timing_valid=not profile,
                    finish_reason=result.finish_reason,batch_id=batch_id,batch_requests=len(group),
                    timing_semantics='amortized batch wall time; NOT individual request latency')
                handle.write(json.dumps(record)+'\n')
            handle.flush()
            batches.append(dict(batch_id=batch_id,requests=len(group),output_tokens=token_count,
                wall_seconds=seconds,cold_wall_seconds=cold,verification_iterations=delta('spec_decode_num_drafts'),
                accepted_draft_tokens=delta('spec_decode_num_accepted_tokens')))
    return batches
