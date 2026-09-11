import json
import argparse
import pytest
from experiments.auf_vllm.compare_outputs import compare
from experiments.handoff_transfer.collect_transfer import main

def test_finish_disagreement_is_reported_even_with_identical_tokens(tmp_path):
    paths=[]
    for mode,finish in [('ar','stop'),('method','length')]:
        path=tmp_path/f'{mode}.jsonl'
        path.write_text(json.dumps({'group_id':'x','timing_valid':True,'output_ids':[1,2],
            'prompt_ids':[3],'output_tokens':2,'wall_seconds':1,'finish_reason':finish})+'\n')
        path.with_suffix('.summary.json').write_text(json.dumps({'timing_valid':True,'contract':{'cap':2048}}))
        paths.append(path)
    result=compare([paths[0]],[paths[1]])
    assert result['exact_matches']==1 and result['finish_matches']==0


def test_main_collector_fails_finish_mismatch_but_retains_report(tmp_path):
    root=tmp_path/'q14';measurements=root/'measurements';measurements.mkdir(parents=True)
    for mode in ('ar','normal','zip','handoff-r56','handoff-five'):
        for repeat in range(3):
            path=measurements/f'{mode}-r{repeat}-w0.jsonl'
            path.write_text('\n'.join(json.dumps({'group_id':str(i),'timing_valid':True,
                'output_ids':[1],'prompt_ids':[2],'output_tokens':1,'wall_seconds':1,
                'finish_reason':'length' if mode=='handoff-five' else 'stop'}) for i in range(128)))
            path.with_suffix('.summary.json').write_text(json.dumps({'timing_valid':True,
                'contract':{'cap':2048,'count':128,'family':'q14','mode':mode,'repeat':repeat,
                            'manifest_sha256':'same','runtime_config':{}}}))
    out=tmp_path/'comparison.json'
    with pytest.raises(AssertionError,match='Full token/finish'):
        main(argparse.Namespace(root=root,out=out))
    report=json.loads(out.read_text())
    assert not report['all_exact']
    assert report['rows']['handoff-five']['exact_token_matches']==[128]*3
    assert report['rows']['handoff-five']['exact_finish_matches']==[0]*3
