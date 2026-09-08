"""Coverage-aware fits: report added value over the same full DDTree reference."""
import json
from pathlib import Path
root=Path('reports')
rows=[]
for p in sorted(root.glob('run-*/lane[0-3]/result.json')):
    cfgpath=p.parent.parent/(p.parent.name+'.json')
    if not cfgpath.exists():continue
    cfg=json.loads(cfgpath.read_text())
    if not cfg.get('decode_variant') or cfg.get('steps',0)<128:continue
    result=json.loads(p.read_text())
    if result.get('status')!='pass':continue
    initial,final=result['validation'][0],result['validation'][-1]
    stages={}
    for stage in ['before','after']:
        s=result[stage]
        stages[stage]={'native_tps':s['native']['tps'],'full_tree_tps':s['tree_reference']['tps'],'student_tps':s['student']['tps'],'original_ratio':s['student']['tps']/s['native']['tps'],'full_tree_ratio':s['student']['tps']/s['tree_reference']['tps'],'student_progress':s['student']['progress'],'full_tree_progress':s['tree_reference']['progress'],'exact_native':s['student']['exact_native'],'requests':s['student']['requests']}
    rows.append(dict(run=p.parents[1].name,lane=p.parent.name,name=cfg['name'],config=cfg,stages=stages,validation_initial=initial,validation_final=final,seconds=result['elapsed_seconds']))
(root/'coverage-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Joint fitting for tree candidate coverage','','Each final checkpoint is tested with DDTree47 and compared against the untouched released drafter with the same tree, as well as original single-path DFlash. Tests are eight development requests, cap512, one timing per stage. These are adaptive single-seed fits, not confirmation. The top-five hinge is a local coverage surrogate; actual accepted progress and TPS decide promotion. The full-model KL and coverage arms are matched; compact arms additionally warm-start earlier fits.','','| Fit | Original TPS | Full DDTree TPS | Student TPS | / Original | / Full DDTree | Before / Full DDTree | Final validation KL | Seconds |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    a=r['stages']['after'];b=r['stages']['before']
    lines.append(f"| {r['name']} | {a['native_tps']:.1f} | {a['full_tree_tps']:.1f} | {a['student_tps']:.1f} | {a['original_ratio']:.3f} | {a['full_tree_ratio']:.3f} | {b['full_tree_ratio']:.3f} | {r['validation_final']['kl']:.3f} | {r['seconds']:.1f} |")
(root/'TREE_COVERAGE.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
