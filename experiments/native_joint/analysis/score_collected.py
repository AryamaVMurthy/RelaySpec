"""Score completed confirmation lanes; withhold summary until full coverage."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
from quality import score,QWEN_EVAL,HERE

ROOT=HERE.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol={'grader':'vendored Qwen2.5-Math parser/grader; HumanEval manifest unit tests','sandbox':'bubblewrap unshare-all; no home/network;5s CPU,8s wall,768MiB address space,1MiB file output','python_source':sha(HERE/'quality.py'),'qwen_source_sha256':{str(p.relative_to(QWEN_EVAL)):sha(p) for p in sorted(QWEN_EVAL.rglob('*.py'))},'dependencies':{'antlr4-python3-runtime':'4.11.1','word2number':'1.1','sympy':'1.12','mpmath':'1.3.0'},'math_policy':'Official extraction and symbolic/numeric equality; parse failures remain incorrect and are logged.','code_policy':'Preserve original function prompt/helpers and use complete returned function when present; execute supplied standard HumanEval tests.','dialogue_policy':'No canonical score. Preserve text, exact agreement and truncation for review.'}
protocol_sha=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
protocol_path=ROOT/'reports/quality-protocol.json'
if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
else:protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
cache_path=ROOT/'reports/quality-cache.jsonl'
cache={}
if cache_path.exists():
    for line in cache_path.read_text().splitlines():
        row=json.loads(line);cache[row['key']]=row['score']
records={r['problem_id']:r for r in json.loads((ROOT/'data/confirmation.json').read_text())['records']}
jobs=json.loads((ROOT/'reports/confirmation-jobs.json').read_text())
outputs=[]
for job in jobs:
    directory=ROOT/f"reports/run-{job['job']}"
    for lane in range(4):
        status=directory/f'lane{lane}-status.json'
        if not status.exists() or json.loads(status.read_text())['exit_code']!=0:continue
        config=json.loads((directory/f'lane{lane}.json').read_text());assert config['phase']=='confirmation'
        folder=directory/f'lane{lane}'
        raw=folder/config['variants'][0]['name']/'evaluation.jsonl'
        outputs.extend(json.loads(line) for line in raw.read_text().splitlines() if json.loads(line)['repeat']==0)
        outputs.extend(json.loads(line) for line in (folder/'ar-quality.jsonl').read_text().splitlines())
pending={};mapping=[]
for output in outputs:
    record=records[output['problem_id']]
    key=hashlib.sha256(json.dumps({'record':record,'completion':output['completion'],'protocol_sha256':protocol_sha},sort_keys=True).encode()).hexdigest()
    mapping.append({'method':output['method'],'problem_id':output['problem_id'],'benchmark':output['benchmark'],'key':key,'capped':output['capped']})
    if key not in cache:pending[key]=(record,output['completion'])
with cache_path.open('a') as stream,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures={pool.submit(score,*payload):key for key,payload in pending.items()}
    for future in concurrent.futures.as_completed(futures):
        key=futures[future];value=future.result();cache[key]=value
        stream.write(json.dumps({'key':key,'score':value})+'\n');stream.flush()
scored=[dict(row,score=cache[row['key']]) for row in mapping]
(ROOT/'reports/quality-scored.json').write_text(json.dumps({'protocol_sha256':protocol_sha,'outputs':scored},indent=2)+'\n')
print(json.dumps({'outputs_scored':len(scored),'unique_scores':len(cache),'new_scores':len(pending)}))
if len(scored)!=640:raise SystemExit(0)
assert len({(r['method'],r['problem_id']) for r in scored})==640
methods=['native','compact_linear','candidate','reference','ar']
rows=[]
for benchmark in ['gsm8k','math500','humaneval','mtbench']:
    for method in methods:
        selected=[r for r in scored if r['benchmark']==benchmark and r['method']==method]
        assert len(selected)==32
        rows.append({'benchmark':benchmark,'method':method,'requests':32,'correct':sum(r['score']['correct'] is True for r in selected) if benchmark!='mtbench' else None,'capped':sum(r['capped'] for r in selected),'failures':[r for r in selected if r['score']['correct'] is False]})
(ROOT/'reports/quality-summary.json').write_text(json.dumps({'status':'complete','protocol_sha256':protocol_sha,'rows':rows},indent=2)+'\n')
lines=['# Confirmation answer and code audit','','Math uses the vendored Qwen2.5-Math extraction/grading rules. HumanEval uses its supplied standard tests in an isolated sandbox. All scores use one deterministic completion per request; repeated identical outputs are not extra samples. Failures and extraction details are retained in quality-scored.json. Dialogue has no ground-truth score and is not claimed equivalent.','','| Method | GSM8K correct /32 | MATH500 correct /32 | HumanEval passed /32 | Dialogue capped /32 |','|---|---:|---:|---:|---:|']
for m in methods:
    get=lambda b:next(r for r in rows if r['method']==m and r['benchmark']==b)
    lines.append(f"| {m} | {get('gsm8k')['correct']} | {get('math500')['correct']} | {get('humaneval')['correct']} | {get('mtbench')['capped']} |")
lines+=['','These small-set scores do not prove distributional equivalence or universal accuracy preservation. BF16 exact AR/native agreements are reported separately in CONFIRMATION.md. Review logged extraction/test failures before attributing a quality difference to a model.']
(ROOT/'reports/CONFIRMATION_QUALITY.md').write_text('\n'.join(lines)+'\n')
