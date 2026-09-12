"""Paired features at audited shared text boundaries for cross-family ZIP fitting."""
import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from experiments.handoff_transfer.cross.alignment import paired_from_cache
from experiments.handoff_transfer.matrix.prepare import digest
from experiments.auf_vllm.runtime_zip.sampling import positions
from experiments.auf_vllm.pilot_data import write


def main(args):
    assert os.environ.get("VLLM_BATCH_INVARIANT")=="1", "Prefix/full feature comparisons require batch-invariant kernels"
    from vllm import LLM, PoolingParams
    source = AutoTokenizer.from_pretrained(args.models/'qwen4-source', local_files_only=True)
    target = AutoTokenizer.from_pretrained(args.models/'llama8-source', local_files_only=True)
    rows = json.loads((args.out/'train.json').read_text())
    manifest_hash = digest(args.out/'train.json')
    labels_path=args.out/'source-label-alignment.json'
    label_contract=json.loads((args.out/'alignment-summary.json').read_text())
    assert label_contract['rollout_sha256']==manifest_hash
    assert label_contract['labels_sha256']==digest(labels_path)
    assert label_contract.get('source_normalizer')==str(source.backend_tokenizer.normalizer), 'Rebuild alignment with native-normalizer provenance before cache reuse'
    generated=json.loads(labels_path.read_text())
    assert len(generated)==len(rows)
    aligned = []
    for row,cached in zip(rows,generated):
        assert cached['group_id']==row['group_id']
        # Include prompt boundaries for initializer fitting. AUF retains the
        # separate generated-token-only alignment from cross.prepare.
        selected = set(positions(len(row['full_ids']), len(row['prompt_token_ids']), row['group_id']))
        item = paired_from_cache(row['full_ids'],len(row['prompt_token_ids']),
                                 source,target,selected,cached)
        item['selected'] = item['blocks']
        assert item['selected'], 'No shared boundaries in stratified sample'
        aligned.append(item)
    del generated
    maximum = max(len(a['source_ids']) for a in aligned)
    config = json.loads((args.models/'qwen4-source/config.json').read_text())
    assert maximum <= config['max_position_embeddings'], 'No source truncation permitted'
    taps = [1,9,17,25,33]
    os.environ['AUF_TARGET_TAPS'] = json.dumps(taps)
    llm = LLM(model=str(args.models/'qwen4-source'), runner='pooling',
        hf_overrides={'architectures':['AUFCaptureQwen3']}, pooler_config={'task':'token_embed'},
        dtype='bfloat16', enforce_eager=True, max_model_len=maximum,
        max_num_seqs=4, max_num_batched_tokens=max(8192,maximum), gpu_memory_utilization=.8,
        enable_chunked_prefill=False, enable_prefix_caching=False)
    options = dict(pooling_task='token_embed', pooling_params=PoolingParams(task='token_embed'), use_tqdm=False)
    for start in range(0,len(rows),4):
        outputs = llm.encode([{'prompt_token_ids':a['source_ids']} for a in aligned[start:start+4]], **options)
        for i, output in enumerate(outputs,start):
            row, item = rows[i], aligned[i]
            y = output.outputs.data.to(dtype=torch.bfloat16,device='cpu')
            assert y.shape == (len(item['source_ids']),12800)
            target_file = args.out/f'features/source/train/{i:05d}.pt'
            provenance = json.loads(target_file.with_suffix('.json').read_text())
            assert provenance['manifest_sha256']==manifest_hash and provenance['sha256']==digest(target_file)
            data = torch.load(target_file,weights_only=True,mmap=True)
            assert data['group_id']==row['group_id'] and data['layers']==[1,8,15,22,29]
            selected = item['selected']
            ti = torch.tensor([b['target_anchor'] for b in selected]);si = torch.tensor([b['source_anchor'] for b in selected])
            if i==0:
                end=int(si[0])+1
                prefix=llm.encode([{'prompt_token_ids':item['source_ids'][:end]}],**options)[0].outputs.data.float().cpu()
                error=float((prefix-y[:end].float()).square().sum()/prefix.square().sum().clamp_min(1e-12))
                write(args.out/'paired/causality.json',dict(passed=error<1e-6,relative_mse=error,threshold=1e-6,
                    prefix_tokens=end,full_tokens=len(item['source_ids']),batch_invariant=True))
                assert error<1e-6, f'Prefix/full feature mismatch: {error}'
            dest=args.out/f'paired/{i:05d}.pt';dest.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(group_id=row['group_id'],x=data['features'][ti].clone(),y=y[si].clone(),
                target_positions=ti,source_positions=si),dest.with_suffix('.tmp'))
            dest.with_suffix('.tmp').replace(dest)
            write(dest.with_suffix('.json'),dict(sha256=digest(dest),manifest_sha256=manifest_hash,
                group_id=row['group_id'],positions=len(selected),target_feature_sha256=provenance['sha256'],
                sampling='original25% target strata intersected with exact source-token prefixes under native source normalization',
                generated_alignment_sha256=label_contract['labels_sha256'],
                alignment_execution='reused audited generated anchors; sampled prompt anchors checked separately',
                weighting='equal total example mass; per-position weight1/retained_count',
                target_taps=[1,8,15,22,29],source_taps=taps,job_id=os.environ.get('SLURM_JOB_ID')))
        print(json.dumps(dict(paired_records=start+len(outputs))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
