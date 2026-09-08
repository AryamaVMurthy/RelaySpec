"""Sandboxed answer/code checks; independent of the frozen inference source."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sysconfig
import tempfile
import uuid

HERE=Path(__file__).resolve().parent
SITE=sysconfig.get_paths()['purelib']
QWEN_EVAL=HERE.parents[2]/'vendor/qwen-math/evaluation'
PRIVATE_DEPS=Path('/tmp/native-quality-deps')


def code_candidate(prompt,completion,entry):
    prompt_blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',prompt,re.S)
    original=prompt_blocks[0] if prompt_blocks else prompt
    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',completion,re.S)
    candidates=blocks or [completion]
    selected=next((s for s in candidates if re.search(r'\bdef\s+'+re.escape(entry)+r'\s*\(',s)),candidates[0])
    full=bool(re.search(r'\bdef\s+'+re.escape(entry)+r'\s*\(',selected))
    code=original+'\n'+selected if full else original+selected
    try:
        ast.parse(code)
    except SyntaxError:
        if full:
            code=selected
        else:
            code=original+'\n'+ '\n'.join('    '+line for line in selected.splitlines())
        ast.parse(code)
    return code


def run_sandbox(payload):
    marker=uuid.uuid4().hex
    wrapper=r'''
import json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
resource.setrlimit(resource.RLIMIT_AS,(768*1024*1024,768*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
resource.setrlimit(resource.RLIMIT_NPROC,(16,16))
sys.path[:0]=['/grader_deps','/qwen_eval','/qwen_eval/latex2sympy','/deps']
payload=json.load(sys.stdin)
if payload['mode']=='math':
    import parser,grader
    prediction=parser.extract_answer(payload['completion'],payload.get('data_name','math'))
    result={'predicted_answer':prediction,'correct':bool(grader.math_equal(prediction,str(payload['answer'])))}
else:
    namespace={'__name__':'candidate'}
    exec(compile(payload['code'],'candidate.py','exec'),namespace)
    exec(compile(payload['test'],'tests.py','exec'),namespace)
    namespace['check'](namespace[payload['entry_point']])
    result={'correct':True}
print(payload['marker']+json.dumps(result))
'''
    payload=dict(payload,marker=marker)
    command=['bwrap','--unshare-all','--new-session','--die-with-parent','--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64','--ro-bind',SITE,'/deps','--ro-bind',str(QWEN_EVAL),'/qwen_eval','--ro-bind',str(PRIVATE_DEPS),'/grader_deps','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp','--clearenv','--setenv','PATH','/usr/bin','--setenv','PYTHONHASHSEED','0','--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1','/usr/bin/python3','-I','-c',wrapper]
    try:
        with tempfile.TemporaryFile(mode='w+') as out, tempfile.TemporaryFile(mode='w+') as err:
            result=subprocess.run(command,input=json.dumps(payload),text=True,stdout=out,stderr=err,timeout=30)
            out.seek(0); stdout=out.read(1024*1024)
            err.seek(0); stderr=err.read(1024*1024)
    except subprocess.TimeoutExpired:
        return {'correct':None,'reason':'sandbox_timeout_unscored'}
    marked=[line[len(marker):] for line in stdout.splitlines() if line.startswith(marker)]
    if result.returncode or not marked:
        if result.returncode not in (0,1) or any(s in stderr for s in ['MemoryError','ModuleNotFoundError','bwrap:']):
            return {'correct':None,'reason':'sandbox_resource_or_infrastructure_failure','returncode':result.returncode,'error':stderr[-1200:]}
        return {'correct':False,'reason':'sandbox_test_or_parse_failure','returncode':result.returncode,'error':stderr[-1200:]}
    return json.loads(marked[-1])


def score(record,completion):
    benchmark=record['benchmark']
    if benchmark in {'gsm8k','math500'}:
        return run_sandbox({'mode':'math','data_name':'gsm8k' if benchmark=='gsm8k' else 'math','completion':completion,'answer':record['answer']})
    if benchmark=='humaneval':
        answer=record['answer']
        try:
            code=code_candidate(record['prompt'],completion,answer['entry_point'])
        except (SyntaxError,ValueError) as error:
            return {'correct':False,'reason':'code_extraction_failure','error':str(error)}
        return run_sandbox({'mode':'code','code':code,'test':answer['test'],'entry_point':answer['entry_point']})
    return {'correct':None,'reason':'No ground-truth dialogue score; inspect agreement, truncation and response text'}


if __name__=='__main__':
    assert run_sandbox({'mode':'math','completion':r'The result is $\boxed{\frac{1}{2}}$.','answer':'0.5'})['correct']
    assert run_sandbox({'mode':'code','code':'def f(x): return x+1','test':'def check(c): assert c(2)==3','entry_point':'f'})['correct']
    assert not run_sandbox({'mode':'code','code':'def f(x): return x','test':'def check(c): assert c(2)==3','entry_point':'f'})['correct']
    assert not run_sandbox({'mode':'code','code':'def f(x):\n while True: pass','test':'def check(c): c(2)','entry_point':'f'})['correct']
    assert run_sandbox({'mode':'code','code':"def f(x):\n import os\n return os.path.exists('/home/aryamavmurthy')",'test':'def check(c): assert c(2) is False','entry_point':'f'})['correct']
    print('Sandbox checks passed: correct/incorrect code, math, timeout, hidden host home')
