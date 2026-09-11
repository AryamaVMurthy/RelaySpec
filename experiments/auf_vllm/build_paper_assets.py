"""Build preliminary manuscript tables only from completed, paired raw outputs."""
import argparse
import json
import shutil
from pathlib import Path
from .compare_outputs import compare
from .plot_decoding_evidence import paths
from .pilot_data import sha,write


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    generated=args.out/'generated';generated.mkdir(exist_ok=True)
    claims=[];sources={}
    def record(files):
        for p in files:
            sources[str(p)]=sha(p)
            if p.suffix=='.jsonl':
                summary=p.with_suffix('.summary.json')
                metadata=json.loads(summary.read_text())
                assert metadata.get('contract',metadata.get('args',{}))['cap']==2048
                sources[str(summary)]=sha(summary)
    table=[r'\begin{tabular}{llrrr}',r'\toprule',r'Target & Objective & Tokens/s & Speedup & Exact/128 \\',r'\midrule']
    for family,label in [('q8','Qwen3-8B'),('llama','Llama-3.2-3B')]:
        root=args.reports/('q8-selected-dev128' if family=='q8' else 'llama-selected-dev128')
        files={m:(paths(root,'ar8' if m=='ar' else m) if family=='q8' else [root/f'{m}-r0-w0.jsonl'])
               for m in ['ar','zip','ce','auf']}
        ar=compare(files['ar'],files['ar'])
        table.append(f"{label} & AR & {ar['ar_tps']:.2f} & 1.00 & reference \\\\")
        for method in ['zip','ce','auf']:
            contract=args.reports/'selected-main-fits-20260911'/family/method/'contract.json'
            fit=json.loads(contract.read_text())
            assert fit['records']==4096 and fit['epochs']==3 and fit['seed']==42
            result=compare(files['ar'],files[method]);assert result['count']==result['exact_matches']==128
            record([contract,*files['ar'],*files[method]])
            table.append(f" & {method.upper()} & {result['method_tps']:.2f} & {result['throughput_ratio']:.2f} & {result['exact_matches']} \\\\")
            claims.append({'id':f'{family}-{method}-development','status':'completed_single_seed_single_timing_development',
                           'result':result,'fit_contract':str(contract),'raw_outputs':[str(p) for p in files[method]],
                           'allowed_scope':'This target, checkpoint, runtime and128 development prompts only'})
        table.append(r'\midrule')
    table[-1]=r'\bottomrule';table.append(r'\end{tabular}')
    (generated/'objective_table.tex').write_text('\n'.join(table)+'\n')
    baseline=paths(args.reports/'q8-selected-dev128','zip')
    table=[r'\begin{tabular}{lrrr}',r'\toprule',r'Qwen adaptation & Tokens/s & Ratio to ZIP & Exact/128 \\',r'\midrule']
    base=compare(baseline,baseline)
    table.append(f"ZIP (earlier timing) & {base['method_tps']:.2f} & 1.000 & 128 \\\\")
    for loss in ['ce','auf']:
        training=args.reports/'draft-lora-main-31450'/f'draft-{loss}-lr2e-5'
        contract=training/'contract.json'
        fit=json.loads(contract.read_text())
        assert fit['records']==4096 and fit['epochs']==3 and fit['seed']==42
        assert fit['draft_lora'] and fit['rank']==32 and fit['original_frozen_targets']
        record([contract,training/'initialization.json',training/'draft-lora.json',training/'offline-validation.json'])
        root=args.reports/'draft-lora-decoding-31450'/f'draft-{loss}'
        row=root/'measurements/development128-cap2048/worker_0/mapped-r0.jsonl'
        result=compare(baseline,[row]);assert result['count']==result['exact_matches']==128
        summary=json.loads(row.with_suffix('.summary.json').read_text())
        assert all(x['passed'] and x['draft_projections']['full_tensor_equality'] and x['draft_projections']['fused_context_kv_verified']
                   for x in summary['attachment_check'])
        record([row,*baseline])
        table.append(f"ZIP + drafter LoRA {loss.upper()} & {result['method_tps']:.2f} & {result['throughput_ratio']:.3f} & 128 \\\\")
        claims.append({'id':f'q8-drafter-lora-{loss}-first-timing','status':'preliminary_repeats_pending',
                       'result':result,'additional_training_epochs':3,'unique_records':4096,
                       'allowed_scope':'No resolved gain or winner from these near-parity single timings'})
    table.extend([r'\bottomrule',r'\end{tabular}'])
    (generated/'lora_table.tex').write_text('\n'.join(table)+'\n')
    figure=args.reports.parent/'figures/qwen_llama_objectives.pdf'
    (args.out/'figures').mkdir(exist_ok=True);shutil.copyfile(figure,args.out/'figures/objectives.pdf')
    record([figure,figure.with_suffix('.json')])
    write(args.out/'claim_evidence.json',{'claims':claims,'source_sha256':sources,
          'not_complete':['Q14','multiple_workloads','timing_repetitions','additional_fitting_seeds','scaling','capacity',
                          'confirmation','final_transformers','full_manuscript'],
          'historical_paper_results_imported':False})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reports',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
