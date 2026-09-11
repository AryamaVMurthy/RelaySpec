import json
from types import SimpleNamespace as NS
import pytest
from experiments.auf_vllm import batch_measurement as bm

class Engine:
    def __init__(self, cold=False, reverse=False):
        self.calls=0; self.cold=cold; self.reverse=reverse
    def collective_rpc(self, name):
        return [dict(jit_events=int(self.cold and self.calls>0),teacher_graph_captures=0)]
    def generate(self, prompts, params, use_tqdm):
        self.calls+=1
        rows=[NS(prompt_token_ids=p['prompt_token_ids'], outputs=[NS(token_ids=[1,2],text='x',finish_reason='length')]) for p in prompts]
        return rows[::-1] if self.reverse else rows
    def counters(self):
        return {'vllm:spec_decode_num_drafts':self.calls*3,'vllm:spec_decode_num_accepted_tokens':self.calls*5}

def selected(n):
    return [(i,dict(prompt_token_ids=[i],group_id=str(i),row_id=i)) for i in range(n)]

def test_batch_totals_and_partial_group(tmp_path, monkeypatch):
    ticks=iter([0,8,10,14]);monkeypatch.setattr(bm.time,'perf_counter',lambda:next(ticks))
    engine=Engine();path=tmp_path/'rows.jsonl'
    batches=bm.measure(engine,selected(6),None,path,4,engine.counters)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert [b['requests'] for b in batches]==[4,2]
    assert sum(r['wall_seconds'] for r in rows)==sum(b['wall_seconds'] for b in batches)==12
    assert sum(r['output_tokens'] for r in rows)==12
    assert sum(b['accepted_draft_tokens'] for b in batches)==10
    assert all('NOT individual' in r['timing_semantics'] for r in rows)

def test_cold_retry_excludes_first_pass_counters(tmp_path,monkeypatch):
    ticks=iter([0,10,20,24]);monkeypatch.setattr(bm.time,'perf_counter',lambda:next(ticks))
    engine=Engine(cold=True)
    batches=bm.measure(engine,selected(4),None,tmp_path/'rows.jsonl',4,engine.counters,profile=True)
    assert engine.calls==2
    assert batches[0]['wall_seconds']==4 and batches[0]['cold_wall_seconds']==10
    assert batches[0]['accepted_draft_tokens']==5
    assert all(not json.loads(x)['timing_valid'] for x in (tmp_path/'rows.jsonl').read_text().splitlines())

def test_reordered_outputs_rejected(tmp_path):
    engine=Engine(reverse=True)
    with pytest.raises(AssertionError):
        bm.measure(engine,selected(4),None,tmp_path/'rows.jsonl',4,engine.counters)

def test_comparison_requires_matched_batch_protocol(tmp_path):
    from experiments.auf_vllm.compare_outputs import compare
    engine=Engine();a=tmp_path/'ar.jsonl';b=tmp_path/'method.jsonl'
    batches=bm.measure(engine,selected(4),None,a,4,engine.counters)
    b.write_text(a.read_text())
    summary=dict(timing_valid=True,contract=dict(cap=2048,workers=1,request_batch_size=4),batch_measurements=batches)
    for path in (a,b):path.with_suffix('.summary.json').write_text(json.dumps(summary))
    result=compare([a],[b])
    assert result['exact_matches']==4 and result['throughput_ratio']==1
    assert result['request_batch_size']==4 and 'not individual latency' in result['ratio_note']
    summary['contract']['request_batch_size']=8
    b.with_suffix('.summary.json').write_text(json.dumps(summary))
    with pytest.raises(AssertionError,match='Unequal serving batch sizes'):compare([a],[b])
    summary['contract']['request_batch_size']=4
    summary['batch_measurements'][0]['wall_seconds']*=2
    b.with_suffix('.summary.json').write_text(json.dumps(summary))
    with pytest.raises(AssertionError):compare([a],[b])
