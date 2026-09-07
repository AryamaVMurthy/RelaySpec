"""Narrow block search around the coarse optima on eight development requests."""
import json
import os
from pathlib import Path
import yaml

root=Path(os.environ['FAMILY_SCALE_CACHE'])
dest=root/'selection'/'fine';dest.mkdir(parents=True,exist_ok=False)
choices=json.loads((root/'selection/blocks/selection.json').read_text())
for choice in choices:
    pair=choice['family'];start=0 if pair=='llama' else 2
    old=yaml.safe_load(Path(f'reports/llama-speed-20260907/lane{start}.yaml').read_text())
    for offset,cfg in enumerate([old,choice['screen_winner']['config']]):
        lane=start+offset;paths=[]
        for block in ([9,10,11] if pair=='llama' else [15,16,17]):
            c=json.loads(json.dumps(cfg));c['run_name']=f'fine-{pair}-lane{lane}-b{block}'
            c['benchmark'].update(block_size=block,max_prompts=8,
                manifest_path='reports/llama-speed-20260907/fresh-manifest.json')
            c['generation']['max_new_tokens']=512
            path=dest/(c['run_name']+'.yaml');path.write_text(yaml.safe_dump(c,sort_keys=False));paths.append(str(path))
        (dest/f'lane{lane}.json').write_text(json.dumps(paths,indent=2))
