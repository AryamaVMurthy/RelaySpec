"""Generate manuscript rows exclusively from complete matrix collector outputs."""
import argparse,json
from pathlib import Path

def escape(value):return str(value).replace('_',r'\_')

def build(paths,out):
    rows=[];seen=set()
    for path in paths:
        result=json.loads(path.read_text())
        assert result['status']=='complete_verified'
        cell=result['cell'];key=(cell['family'],cell['kind'],cell['objective'])
        assert key not in seen;seen.add(key)
        fit=result['fit'];assert fit['optimizer_steps']==2000 and fit['anchors_per_example']==512
        repeats=result['repeats'];assert len(repeats)==3
        assert all(r['count']==r['exact_matches']==r['finish_matches']==128 for r in repeats)
        rows.append((key,result,path))
    if not rows:raise ValueError('No complete benchmarks; do not generate a result table')
    lines=[r'\begin{tabular}{lllrrrr}',r'\toprule',r'Target & Interface & Loss & Tokens/s & /AR & /RelaySpec & /ZIP \\',r'\midrule']
    provenance=[]
    for key,r,path in sorted(rows,key=lambda x:x[0]):
        columns=[*(escape(x) for x in key),f"{r['mean_tps']:.1f}",f"{r['mean_paired_ar_speedup']:.2f}",f"{r['ratio_to_controls']['normal']:.3f}",f"{r['ratio_to_controls']['zip']:.3f}"]
        lines.append(' & '.join(columns)+r' \\')
        provenance.append({'cell':key,'source':str(path),'checkpoint_sha256':r['verification']['export_sha256']})
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'matrix_table.tex').write_text('\n'.join(lines)+'\n')
    (out/'matrix_table_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.results,a.out)
