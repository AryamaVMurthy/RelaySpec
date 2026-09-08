"""Keep reserved inference confirmation separate from adaptive development."""
import hashlib
import json
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_signature(spec, source_directory, target, draft, source_commit):
    return {
        'eval_manifest_sha256': file_sha(spec['eval_manifest']),
        'variants': spec['variants'],
        'reference_variant': spec.get('reference_variant'),
        'candidate_checkpoint': spec.get('candidate_checkpoint'),
        'quantization': spec.get('quantization'),
        'include_compact_linear': spec.get('include_compact_linear', False),
        'include_ar_quality': spec.get('include_ar_quality', False),
        'required_runtime': spec.get('required_runtime'),
        'output_cap': spec['output_cap'],
        'repeats': spec.get('repeats', 1),
        'allow_bf16_reduction': spec.get('allow_bf16_reduction', True),
        'target': target, 'draft': draft, 'source_commit': source_commit,
        'local_launch_sha256': {p.name:file_sha(p) for p in sorted(Path(source_directory).glob('*.sbatch'))},
        'local_source_sha256': {p.name:file_sha(p) for p in sorted(Path(source_directory).glob('*.py'))},
    }


def validate_protocol(spec, source_directory, target, draft, source_commit):
    if spec.get('required_runtime'):
        import torch, transformers
        actual={'torch':torch.__version__,'transformers':transformers.__version__}
        if actual != spec['required_runtime']:
            raise ValueError('Runtime differs from frozen versions')
    manifest=json.loads(Path(spec['eval_manifest']).read_text())
    phase=spec.get('phase','development')
    if phase not in {'development','confirmation'}:
        raise ValueError('Unknown evaluation phase')
    reserved=manifest.get('provenance',{}).get('selection')=='confirmation'
    if reserved != (phase=='confirmation'):
        raise ValueError('Reserved manifest requires explicit frozen confirmation phase')
    if phase=='development':
        return {'phase':phase, 'eval_manifest_sha256':file_sha(spec['eval_manifest'])}
    if not spec.get('frozen_selection') or not spec.get('frozen_selection_sha256'):
        raise ValueError('Freeze selection and its hash before using reserved requests')
    if file_sha(spec['frozen_selection']) != spec['frozen_selection_sha256']:
        raise ValueError('Frozen selection changed')
    frozen=json.loads(Path(spec['frozen_selection']).read_text())
    if frozen['protocol'] != protocol_signature(spec, source_directory, target, draft, source_commit):
        raise ValueError('Confirmation differs from frozen models, decoder, precision or timing protocol')
    ids=[r['problem_id'] for r in manifest['records']]
    if len(ids)!=128 or len(set(ids))!=128 or sorted(ids)!=sorted(frozen['problem_ids']):
        raise ValueError('Confirmation must cover the frozen128 unique requests')
    if spec['output_cap']!=2048 or spec.get('repeats',1)<2:
        raise ValueError('Confirmation requires cap2048 and repeated timing')
    return {'phase':phase,'frozen_selection_sha256':spec['frozen_selection_sha256'],'protocol':frozen['protocol']}
