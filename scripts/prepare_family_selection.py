"""Prepare follow-up comparisons using completed, correctness-gated screens."""
import argparse
import json
import os
from pathlib import Path
import yaml

p = argparse.ArgumentParser()
p.add_argument('--stage',choices=['blocks','validate','final'],required=True)
a = p.parse_args()
root = Path(os.environ['FAMILY_SCALE_CACHE'])
report = Path('reports/family-scale-20260907')
generated = root/'selection'/a.stage
generated.mkdir(parents=True,exist_ok=False)
results = []
if a.stage == 'blocks':
    paths = [root/f'screen-{n}'/f'lane{lane}'/'results.json'
             for n in [4096,8192,16384] for lane in range(4)]
    paths += [root/'refine-screen'/f'lane{lane}'/'results.json' for lane in range(4)]
else:
    previous='blocks' if a.stage=='validate' else 'validate'
    paths = [root/previous/f'lane{lane}'/'results.json' for lane in range(4)]
    if a.stage=='final':
        paths += [root/'fine'/f'lane{lane}'/'results.json' for lane in range(4)]
for path in paths:
    assert (path.parent/'COMPLETE').exists(),path
    for result in json.loads(path.read_text()):
        if not result['exact_gate']:
            continue
        cfg = yaml.safe_load(Path(result['config']).read_text())
        method = next(k for k in result['methods'] if k != 'native_ar')
        results.append({'config':cfg,'result':result,
                        'tps':result['methods'][method]['decode_tokens_per_second']})
choices = []
for family,base_lane in [('llama',0),('cross',2)]:
    baseline = yaml.safe_load(Path(f'reports/llama-speed-20260907/lane{base_lane}.yaml').read_text())
    candidates = [r for r in results if r['config']['target']['id']==baseline['target']['id']]
    if a.stage == 'blocks':
        candidates = [r for r in candidates if '/family-scale-20260907/fits/' in r['config']['relay_probe']['checkpoint_path']]
        winner = max(candidates,key=lambda r:r['tps'])
        configs = [baseline,winner['config']]
    elif a.stage=='validate':
        leaders=sorted(candidates,key=lambda r:r['tps'],reverse=True)[:2]
        assert len(leaders)==2
        winner=leaders[0]
        configs=[r['config'] for r in leaders]
    else:
        winner = max(candidates,key=lambda r:r['tps'])
        configs = [baseline,winner['config']]
    choices.append({'family':family,'screen_winner':winner})
    for offset,cfg in enumerate(configs):
        lane=base_lane+offset; lists=[]
        blocks = ([6,8,10,12,16,24] if family=='llama' else [8,12,16,20,24,32]) if a.stage=='blocks' else [cfg['benchmark']['block_size']]
        for block in blocks:
            c=json.loads(json.dumps(cfg))
            c['run_name']=f'{a.stage}-{family}-lane{lane}-b{block}'
            manifest={'blocks':report/'pilot-manifest.json','validate':Path('reports/llama-speed-20260907/fresh-manifest.json'),
                      'final':report/'final-manifest.json'}[a.stage]
            c['benchmark'].update(block_size=block,max_prompts={'blocks':2,'validate':8,'final':16}[a.stage],
                manifest_path=str(manifest))
            c['generation']['max_new_tokens']={'blocks':256,'validate':512,'final':1024}[a.stage]
            path=generated/(c['run_name']+'.yaml')
            path.write_text(yaml.safe_dump(c,sort_keys=False));lists.append(str(path))
        (generated/f'lane{lane}.json').write_text(json.dumps(lists,indent=2))
(generated/'selection.json').write_text(json.dumps(choices,indent=2))
