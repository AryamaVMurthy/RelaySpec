"""Matched uniform-CE ablation; only the AUF support multiplier changes."""
import importlib.util,types
from pathlib import Path
_root=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('_handoff_transfer_base',_root/'port/model.py')
_base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_base)
for _name in dir(_base):
    if not _name.startswith('_'):globals()[_name]=getattr(_base,_name)

_source=(_root/'vendor/code/objectives.py').read_text()
assert _source.count('loss_weights=weight_mask*support')==1
_ce_source=_source.replace('loss_weights=weight_mask*support','loss_weights=weight_mask')
_ce=types.ModuleType('handoff_uniform_ce');exec(compile(_ce_source,str(_root/'vendor/code/objectives.py')+' [uniform CE ablation]','exec'),_ce.__dict__)

def build(kind):
    assert kind=='fusion_r56'
    wrapper,cfg=_base.build(kind)
    _ce.configure(wrapper,'auf')
    if int(os.environ.get('LOCAL_RANK','0'))==0:
        path=R/f'model-contract-{kind}.json'
        contract=json.loads(path.read_text());contract.update(objective='uniform valid-position CE',
            source_change='loss_weights=weight_mask*support -> loss_weights=weight_mask',
            objective_source_sha256=hashlib.sha256(_ce_source.encode()).hexdigest())
        put(path,contract)
    return wrapper,cfg
