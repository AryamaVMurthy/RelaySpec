"""Audit real rollout labels before any heterogeneous-vocabulary GPU fitting."""
import argparse, json, time
from pathlib import Path
from transformers import AutoTokenizer
from experiments.handoff_transfer.cross.alignment import aligned_blocks
from experiments.auf_vllm.pilot_data import write, sha

def main(a):
    source=AutoTokenizer.from_pretrained(a.models/'qwen4-source',local_files_only=True)
    target=AutoTokenizer.from_pretrained(a.models/'llama8-source',local_files_only=True)
    rows=json.loads((a.out/'train.json').read_text());results=[];start=time.perf_counter()
    for row in rows:
        result=aligned_blocks(row['full_ids'],len(row['prompt_token_ids']),source,target)
        result['group_id']=row['group_id'];results.append(result)
        print(json.dumps({'group_id':row['group_id'],'eligible_anchors':len(result['blocks']),
                          'seconds':time.perf_counter()-start}),flush=True)
    write(a.out/'source-label-alignment.json',results)
    write(a.out/'alignment-summary.json',{'records':len(rows),'eligible_anchors':[len(r['blocks']) for r in results],
          'rollout_sha256':sha(a.out/'train.json'),'labels_sha256':sha(a.out/'source-label-alignment.json'),
          'seconds':time.perf_counter()-start,'status':'labels_audited_training_and_inference_not_yet_validated',
          'source':str(a.models/'qwen4-source'),'target':str(a.models/'llama8-source'),
          'protocol':'exact source-token prefix equality under native source text normalization; target context strictly before anchor',
          'source_normalizer':str(source.backend_tokenizer.normalizer),
          'vocabulary_units':'source labels; target conditioning positions; not target-token AUF'})
    assert all(r['blocks'] for r in results), 'A rollout has no valid source-label anchors'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--models',type=Path,required=True);main(p.parse_args())
