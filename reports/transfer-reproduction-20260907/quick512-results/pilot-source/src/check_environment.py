import sys,json,importlib.metadata as m,platform
from paths import PACKAGE
expected=json.loads((PACKAGE/'provenance/environment.json').read_text())
assert sys.version_info[:3]==(3,12,13),sys.version
bad={k:(v,m.version(k)) for k,v in expected['packages'].items() if m.version(k)!=v}
assert not bad,bad
print(json.dumps({'packages_match':True,'python':sys.version,'platform':platform.platform()}))
