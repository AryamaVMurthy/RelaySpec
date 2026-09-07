"""Choose each family's strongest new fit for LR and longer-epoch follow-ups."""
import json
import os
from pathlib import Path
import yaml

root=Path(os.environ['FAMILY_SCALE_CACHE'])
prepared=root/'refinement'; prepared.mkdir(exist_ok=False)
rows=[]
for n in [4096,8192,16384]:
    for lane in range(4):
        path=root/f'screen-{n}'/f'lane{lane}'/'results.json'
        assert (path.parent/'COMPLETE').exists()
        for result in json.loads(path.read_text()):
            cfg=yaml.safe_load(Path(result['config']).read_text())
            checkpoint=Path(cfg['relay_probe']['checkpoint_path'])
            if result['exact_gate'] and '/family-scale-20260907/fits/' in str(checkpoint):
                method=next(k for k in result['methods'] if k!='native_ar')
                rows.append((result['methods'][method]['decode_tokens_per_second'],cfg,checkpoint))
choices=[]
for pair,start in [('llama',0),('cross',2)]:
    candidates=[r for r in rows if f'-{pair}-' in str(r[2])]
    tps,cfg,checkpoint=max(candidates,key=lambda r:r[0])
    trial=json.loads((checkpoint.parent/'trial.json').read_text())
    choices.append({'pair':pair,'selected_checkpoint':str(checkpoint),'screen_tps':tps,'trial':trial})
    for offset,(lr,epochs,label) in enumerate([(.0003,12,'low-lr'),(.001,24,'long')]):
        lane=start+offset; dest=root/'fits'/f'refine-{pair}-{label}'
        command=['scripts/fit_family_scaling.py','--pair',pair,'--objective',trial['objective'],
            '--records',str(trial['records']),'--epochs',str(epochs),'--lr',str(lr),'--output',str(dest)]
        (prepared/f'fit-lane{lane}.json').write_text(json.dumps(command))
        configs=[]
        for epoch in sorted(set([3,6,12,epochs])):
            c=json.loads(json.dumps(cfg)); c['run_name']=f'refine-{pair}-{label}-e{epoch}'
            c['relay_probe']['checkpoint_path']=str(dest/f'epoch-{epoch}.pt')
            path=prepared/(c['run_name']+'.yaml');path.write_text(yaml.safe_dump(c,sort_keys=False));configs.append(str(path))
        (prepared/f'screen-lane{lane}.json').write_text(json.dumps(configs,indent=2))
(prepared/'selection.json').write_text(json.dumps(choices,indent=2))
