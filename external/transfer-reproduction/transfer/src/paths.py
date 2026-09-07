from pathlib import Path
import os,json
PACKAGE=Path(__file__).resolve().parents[1]
WORK=Path(os.environ.get('TRANSFER_WORK',str(PACKAGE/'work'))).resolve()
def model(size,kind='target'):return WORK/'models'/f'{size}b'/kind
def put(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.part');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def code_hash():
 import hashlib
 h=hashlib.sha256()
 for p in sorted((PACKAGE/'src').glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
 return h.hexdigest()
