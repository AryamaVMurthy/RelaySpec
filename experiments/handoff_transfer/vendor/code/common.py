import json,sys,hashlib,os
from pathlib import Path
CODE=Path(__file__).resolve().parent
R=Path(os.environ['AUF_WORKDIR']).resolve()
BASE=R.parents[1]
OLD=R.parent/'dflash_lora_pilot_20260906'
PAIR=Path(os.environ['AUF_MODELS']).resolve()
SOURCE=R.parent/'dflash_lora_objective_20260907'
PREV=R
UP=Path(os.environ['SPECFORGE_ROOT']).resolve()
sys.path.append(str(CODE))
DOMAIN=os.environ['AUF_DOMAIN']
assert DOMAIN in ['math','kicad','nanocoder']
def put(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_upstream():
 """Import unchanged concrete upstream modules without unrelated strategy imports."""
 import types
 sys.path.insert(0,str(UP))
 import specforge.modeling
 name='specforge.modeling.draft'
 if name not in sys.modules:
  package=types.ModuleType(name);package.__path__=[str(UP/'specforge/modeling/draft')];sys.modules[name]=package
 from specforge.modeling.draft.dflash import DFlashDraftModel
 from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel,create_dflash_sdpa_mask
 return DFlashDraftModel,OnlineDFlashModel,create_dflash_sdpa_mask
