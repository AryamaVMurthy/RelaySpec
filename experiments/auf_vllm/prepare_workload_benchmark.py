"""Materialize the frozen workload view for an existing same-family runner."""
import argparse
import json
from pathlib import Path
from .pilot_data import rows,write,sha


def main(args):
    family='llama' if args.family=='llama' else 'qwen'
    selected=rows(args.manifest/args.workload/f'{args.split}.jsonl')
    warm=rows(args.manifest/args.workload/'warmup.jsonl')
    assert len(selected)==128 and len(warm)==4
    assert len({r['group_id'] for r in selected+warm})==132
    for name,data in [('eval',selected),('warmup',warm)]:
        converted=[]
        for row in data:
            ids=row['prompt_ids_by_family'][family]
            assert len(ids)+2048+16<=5120
            converted.append(dict(row,prompt_token_ids=ids))
        dest=args.out/f'{name}.json'
        if dest.exists():assert json.loads(dest.read_text())==converted
        else:write(dest,converted)
    write(args.out/'provenance.json',{'source_sha256':sha(args.manifest/args.workload/f'{args.split}.jsonl'),
          'workload':args.workload,'split':args.split,'family':args.family,
          'count':128,'cap':2048,'natural_eos':True,'source_provenance_sha256':sha(args.manifest/'provenance.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--family',choices=['q8','q14','llama'],required=True)
    p.add_argument('--workload',choices=['math','gsm','code','chat'],required=True)
    p.add_argument('--split',choices=['development','confirmation'],required=True)
    main(p.parse_args())
