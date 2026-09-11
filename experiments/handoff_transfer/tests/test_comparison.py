import json
from experiments.auf_vllm.compare_outputs import compare

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
