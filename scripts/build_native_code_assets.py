"""Recompute native code scores and export a compact, scoped paper paragraph."""
import json
import runpy
from pathlib import Path

runpy.run_path('scripts/score_native_code_study.py', run_name='__main__')
d = json.loads(Path('reports/autoresearch-20260907/native-code-summary.json').read_text())
methods = d['results']
if any(r['requests'] != 32 or r['code_matches'] != 32 or r['token_matches'] != 32 or r['capped'] != 0 or r['passed'] != 28 for r in methods.values()):
    raise ValueError('Native code result differs from paper statement')
two = d['throughput']['methods']['relay_two']
one = d['throughput']['methods']['relay_one']
svd = d['throughput']['methods']['relay_svd1536']
low, high = two['throughput_ci95']
text = r'''\paragraph{Executable-code development check.}
A separate fixed set of 32 MBPP tasks compares all five frozen native
variants at a 1,024-token cap. These tasks exclude the earlier autoresearch
MBPP IDs but appear in historical evaluations, so they are development
examples. Every variant produces the same tokens and extracted code,
passes 28/32 published base-test suites and has no cap-length outputs.
The four failures are shared. Tests execute in an isolated Python process
using the first Python code block, without EvalPlus extended tests.
'''
text += (f"Two layers retain {100*two['throughput_ratio']:.1f}\\% "
         f"[{100*low:.1f},{100*high:.1f}]\\% of released-native throughput, "
         f"versus {100*one['throughput_ratio']:.1f}\\% for one layer and "
         f"{100*svd['throughput_ratio']:.1f}\\% for rank-1,536 SVD.\n")
text += r'''This supports observed functional agreement alongside a workload-dependent
efficiency penalty. The conservative paired accuracy interval is
$[-12.80,+12.80]$ percentage points, not tight quality noninferiority.
'''
Path('paper/iclr2027/generated/native_code_paragraph.tex').write_text(text)
