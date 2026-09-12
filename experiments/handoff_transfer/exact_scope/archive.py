"""Preserve compact inference interfaces and experiment evidence off scratch."""
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def extract_interface(source,destination):
    from safetensors import safe_open
    from safetensors.torch import save_file
    destination=Path(destination)
    temporary=destination.with_suffix(destination.suffix+'.tmp')
    with safe_open(source,framework='pt',device='cpu') as weights:
        save_file({'fc.weight':weights.get_tensor('fc.weight')},temporary)
    temporary.replace(destination)


def archive(audit,root,destination,config_root):
    if audit['status']!='complete':raise ValueError('Final archive requires a complete audit')
    root,destination=Path(root),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    entries=[]

    def record(path,source):
        entries.append(dict(path=str(path.relative_to(destination)),source=str(source),
                            bytes=path.stat().st_size,sha256=digest(path)))

    def copy(source,relative):
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.tmp')
        shutil.copyfile(source,temporary)
        if digest(source)!=digest(temporary):raise ValueError(f'Archive copy differs: {source}')
        temporary.replace(target);record(target,source)

    def json_tree(source,relative):
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.jsonl'):
                copy(path,Path(relative)/path.relative_to(source))

    for family in ('q8','llama','cross'):
        tasks=json.loads((config_root/f'{family}-tasks.json').read_text())
        tasks += [dict(kind='feature',objective=o,lr=.001) for o in ('feature_ce','forward_kl','reverse_kl')]
        for task in tasks:
            kind,obj,lr=task['kind'],task['objective'],task['lr']
            fit=(root/'feature-objectives-e1'/family/obj if kind=='feature' else
                 root/'matrix32e1b8'/family/f'{kind}-{obj}-lr{lr}')
            export=fit/'export' if kind=='feature' else fit/f'exports/steps-512/{kind}'
            relative=Path('interfaces')/family/f'{kind}-{obj}-lr{lr}'
            dest=destination/relative;dest.mkdir(parents=True,exist_ok=True)
            # All other deployment weights were verified frozen by the fit and
            # benchmark checks. Preserve only the changed fusion interface.
            extract_interface(export/'model.safetensors',dest/'interface.safetensors')
            record(dest/'interface.safetensors',export/'model.safetensors')
            copy(export/'config.json',relative/'config.json')
            if kind=='feature':
                for name in ('summary.json','contract.json'):copy(fit/name,relative/name)
            else:
                module=fit/f'modules/steps-512/{kind}'
                for name in ('summary.json','verification.json'):copy(module/name,relative/name)
                copy(fit/'transfer.json',relative/'transfer.json')
                copy(fit/f'model-contract-{kind}.json',relative/'model-contract.json')
                parameters='parameters.pt' if family=='cross' else 'final.pt'
                copy(module/parameters,relative/'trainable-parameters.pt')
                for path in sorted((fit/'logs').glob('gpu-*.csv')):copy(path,relative/path.name)
        json_tree(root/'exact32e1b8-results'/family,Path('evaluations')/family)
        initial=root/'cross-full4096/initializers-e1' if family=='cross' else root/family
        copy(initial/'transfer.json',Path('initializers')/family/'transfer.json')
        transfer=json.loads((initial/'transfer.json').read_text())
        for index,(name,expected) in enumerate(sorted(transfer['feature_manifests'].items())):
            source=Path(name)
            if digest(source)!=expected:raise ValueError(f'Calibration manifest changed: {source}')
            copy(source,Path('calibration')/family/f'manifest-{index:03d}.json')
        for name,export in [('original',initial/'normal/export'),('zip',Path(transfer['base_export']))]:
            relative=Path('initializers')/family/name
            dest=destination/relative;dest.mkdir(parents=True,exist_ok=True)
            extract_interface(export/'model.safetensors',dest/'interface.safetensors')
            record(dest/'interface.safetensors',export/'model.safetensors')
            copy(export/'config.json',relative/'config.json')
    for chunk in range(64):
        source=root/f'cross-full4096/chunks/{chunk:05d}'
        for name in ('train.json','alignment-summary.json','paired/causality.json'):
            copy(source/name,Path('calibration/cross/chunks')/f'{chunk:05d}'/name)
    json_tree(root/'transformers-e1b8/q8',Path('transformers/q8'))
    for path in sorted((root/'transformers-e1b8/q8').rglob('gpu-*.csv')):
        copy(path,Path('transformers/q8')/path.relative_to(root/'transformers-e1b8/q8'))
    # Device identity stays in each benchmark summary. These CSVs describe
    # whole jobs, including setup and warmup, rather than isolated timed passes.
    jobs={row['benchmark_job_id'] for row in audit['comparisons']}
    for cell in audit['cells']:
        summary=cell['training_summary']
        if 'job_id' in summary:jobs.add(summary['job_id'])
    for job in sorted(map(str,jobs)):
        copy(root.parent/f'exact32-source-{job}/gpu.csv',Path('telemetry')/f'job-{job}.csv')
    for path in sorted(config_root.glob('*')):
        if path.suffix in ('.json','.py','.sbatch','.md'):copy(path,Path('configuration')/path.name)
    manifest=dict(status='complete',files=entries,total_bytes=sum(e['bytes'] for e in entries),
                  reconstruction='Replace fc.weight in the matching frozen drafter with interface.safetensors '
                  'and use the archived config plus pinned RelaySpec runtime. Trainable parameter files retain '
                  'the unmerged token-fit architectures. Feature-fit interfaces are saved at deployment precision; '
                  'full frozen language-model weights and feature caches are not duplicated here.')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(path=str(destination),files=len(entries),bytes=manifest['total_bytes'])
