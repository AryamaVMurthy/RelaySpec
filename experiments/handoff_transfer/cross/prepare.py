"""Audit real rollout labels before any heterogeneous-vocabulary GPU fitting."""
import argparse, json, time, os, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer
from experiments.handoff_transfer.cross.alignment import aligned_blocks
from experiments.auf_vllm.pilot_data import write, sha

def initialize_tokenizers(models):
    global source,target
    source=AutoTokenizer.from_pretrained(models/'qwen4-source',local_files_only=True)
    target=AutoTokenizer.from_pretrained(models/'llama8-source',local_files_only=True)


def align_row(row):
    result=aligned_blocks(row['full_ids'],len(row['prompt_token_ids']),source,target)
    result['group_id']=row['group_id']
    return result


def main(a):
    initialize_tokenizers(a.models)
    rows=json.loads((a.out/'train.json').read_text());results=[];start=time.perf_counter()
    workers=min(a.workers,max(1,len(rows)))
    def retain(aligned):
        for result in aligned:
            results.append(result)
            print(json.dumps({'group_id':result['group_id'],'eligible_anchors':len(result['blocks']),
                              'seconds':time.perf_counter()-start}),flush=True)
    if workers==1:
        retain(map(align_row,rows))
    else:
        # Each record is independent. Ordered map preserves byte-identical
        # labels and record order; spawning avoids inherited tokenizer threads.
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_tokenizers,initargs=(a.models,)) as pool:
            retain(pool.map(align_row,rows,chunksize=1))
    write(a.out/'source-label-alignment.json',results)
    write(a.out/'alignment-summary.json',{'records':len(rows),'eligible_anchors':[len(r['blocks']) for r in results],
          'rollout_sha256':sha(a.out/'train.json'),'labels_sha256':sha(a.out/'source-label-alignment.json'),
          'seconds':time.perf_counter()-start,'alignment_workers':workers,'status':'labels_audited_training_and_inference_not_yet_validated',
          'source':str(a.models/'qwen4-source'),'target':str(a.models/'llama8-source'),
          'protocol':'exact source-token prefix equality under native source text normalization; target context strictly before anchor',
          'source_normalizer':str(source.backend_tokenizer.normalizer),
          'unicode_tail_records':[dict(group_id=r['group_id'],**r['unicode_tail'])
                                  for r in results if 'unicode_tail' in r],
          'vocabulary_units':'source labels; target conditioning positions; not target-token AUF'})
    assert all(r['blocks'] for r in results), 'A rollout has no valid source-label anchors'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,5),default=min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))))
    main(p.parse_args())
