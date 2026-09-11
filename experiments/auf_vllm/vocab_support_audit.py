"""Bounded CPU audit of exact AUF support under the archived token bridge."""
import argparse
import collections
import json
import os
from pathlib import Path
from .pilot_data import sha,write


def main(args):
    from transformers import AutoTokenizer
    source=AutoTokenizer.from_pretrained(args.models/'qwen4-source',local_files_only=True)
    target=AutoTokenizer.from_pretrained(args.models/'llama8-source',local_files_only=True)
    bridge={int(k):v for k,v in json.loads(args.bridge.read_text()).items()}
    lengths=collections.Counter()
    singleton_destinations=set()
    examples={'empty':[],'multiple':[],'lookup_differs_from_reencode':[]}
    singleton_count=0
    mismatches=0
    # This measures vocabulary/retokenization structure, not model probabilities.
    for sid in sorted(set(source.get_vocab().values())):
        text=source.decode([sid],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        encoded=target.encode(text,add_special_tokens=False)
        kind='empty' if len(encoded)==0 else 'single' if len(encoded)==1 else 'multiple'
        lengths[kind]+=1
        if len(encoded)==1:
            singleton_count+=1
            singleton_destinations.add(encoded[0])
        if kind in examples and len(examples[kind])<6:
            examples[kind].append({'source_id':sid,'text':text,'target_ids':encoded})
        if sid in bridge and [bridge[sid]]!=encoded:
            mismatches+=1
            if len(examples['lookup_differs_from_reencode'])<6:
                examples['lookup_differs_from_reencode'].append({'source_id':sid,'lookup_id':bridge[sid],'text':text,'reencoded_ids':encoded})
    probes=[]
    samples=['Hello, world!','def f(x):\n    return x + 1\n','café 日本語 😀','  12.5\n\n',
             'The answer is 42.','\\boxed{\\frac{1}{2}}','a\u0301','你好，世界！']
    for text in samples:
        ids=source.encode(text,add_special_tokens=False)
        decoded=source.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        expected=target.encode(decoded,add_special_tokens=False)
        mapped=[bridge.get(i) for i in ids]
        probes.append({'text':text,'source_ids':ids,'target_reencoded_ids':expected,
                       'all_lookup_covered':all(i is not None for i in mapped),
                       'lookup_ids':mapped,'lookup_matches_reencode':mapped==expected,
                       'length_preserved':len(ids)==len(expected)})
    target_eos=target.eos_token_id
    target_vocab=set(target.get_vocab().values())
    missing=target_vocab-singleton_destinations
    report={'status':'dynamic_tokenizer_and_support_audit','job_id':os.environ.get('SLURM_JOB_ID'),
            'gpu_used':False,'target_adapters':None,'shared_vocab':source.get_vocab()==target.get_vocab(),
            'bridge_sha256':sha(args.bridge),'source_config_sha256':sha(args.models/'qwen4-source/config.json'),
            'target_config_sha256':sha(args.models/'llama8-source/config.json'),
            'source_token_count':len(source.get_vocab()),'target_token_count':len(target.get_vocab()),
            'single_source_token_retokenization_counts':dict(lengths),'singleton_target_coverage':len(singleton_destinations),
            'singleton_target_missing':len(missing),'target_eos':target_eos,'target_eos_has_singleton_support':target_eos in singleton_destinations,
            'lookup_entries':len(bridge),'lookup_distinct_destinations':len(set(bridge.values())),
            'lookup_single_token_reencoding_mismatches':mismatches,
            'uniform_source_distribution_singleton_mass':singleton_count/len(source.get_vocab()),
            'uniform_mass_note':'synthetic uniform vocabulary mass, not a measured drafter distribution',
            'examples':examples,'probes':probes,
            'exact_per_target_position_auf_supported_by_existing_bridge':False,
            'reason':'Variable-length retokenization is not a per-position target-vocabulary distribution. Restricting and renormalizing singleton tokens changes proposals and leaves missing target labels with zero probability.',
            'scope':'This audits the existing bridge only; it is not an impossibility result for a new vocabulary-aware architecture.'}
    assert not report['shared_vocab']
    assert lengths['multiple']>0 and len(missing)>0
    write(args.out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['examples','probes']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--bridge',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
