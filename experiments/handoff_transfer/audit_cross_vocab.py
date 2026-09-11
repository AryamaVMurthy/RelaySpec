"""Audit the heterogeneous-tokenization boundary before a cross-family AUF port."""
import argparse,json,hashlib,statistics
from pathlib import Path
from transformers import AutoTokenizer

def main(a):
    source=AutoTokenizer.from_pretrained(a.source,local_files_only=True)
    target=AutoTokenizer.from_pretrained(a.target,local_files_only=True)
    sv,tv=source.get_vocab(),target.get_vocab()
    sc=json.loads((a.source/'config.json').read_text());tc=json.loads((a.target/'config.json').read_text())
    same_id=[{'token':t,'source_id':i,'target_id':tv[t]} for t,i in sv.items() if t in tv and i!=tv[t]][:12]
    cases=['The answer is 42.','def foo(x):\n    return x + 1','你好，世界！ Café π = 3.14159','\\boxed{12345}']
    cases += [json.loads(line)['problem'] for line in a.manifest.read_text().splitlines()[:128]]
    results=[]
    for text in cases:
        s=source(text,add_special_tokens=False,return_offsets_mapping=True)
        t=target(text,add_special_tokens=False,return_offsets_mapping=True)
        se={end for start,end in s['offset_mapping'] if end>start}
        te={end for start,end in t['offset_mapping'] if end>start}
        results.append({'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'source_length':len(s['input_ids']),
            'target_length':len(t['input_ids']),'identical_ids':s['input_ids']==t['input_ids'],
            'source_boundaries':len(se),'target_boundaries':len(te),'common_boundaries':len(se&te),
            'source_prefix_ids':s['input_ids'][:12],'target_prefix_ids':t['input_ids'][:12]})
    result={'source':str(a.source),'target':str(a.target),'source_config_vocab_size':sc['vocab_size'],
        'target_config_vocab_size':tc['vocab_size'],'source_tokenizer_vocab_size':len(sv),'target_tokenizer_vocab_size':len(tv),
        'shared_token_string_count':len(sv.keys()&tv.keys()),'identical_full_vocabulary':sv==tv,
        'different_ids_for_shared_tokens':same_id,'cases':results,
        'cases_with_different_token_counts':sum(r['source_length']!=r['target_length'] for r in results),
        'source_boundary_alignment_fraction':sum(r['common_boundaries'] for r in results)/sum(r['source_boundaries'] for r in results),
        'direct_handoff_port_supported':sv==tv,
        'conclusion':'A linear hidden mapper does not map token IDs. Source-token AUF prefix support and target-token acceptance have different position units; the unchanged handoff OnlineDFlash supervision is not a heterogeneous-vocabulary bridge.',
        'extension_required':'Align causal conditioning/text prefixes and anchor positions, train on source-vocabulary labels, and verify re-tokenized proposals with the target. Report this as a separate bridge extension, not identical handoff training.',
        'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in [a.source,a.target] for p in [root/'config.json',root/'tokenizer.json']}}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','hashes','different_ids_for_shared_tokens']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
