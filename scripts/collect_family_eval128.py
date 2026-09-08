"""Persistently collect the Slurm array; generate an explicit completion report."""
import json,subprocess,time,sys
from pathlib import Path
report=Path('reports/family-eval128-20260908');artifacts=report/'artifacts'
while True:
 try:
  subprocess.run(['rsync','-a','--include=*/','--include=*.json','--include=*.jsonl','--include=*.yaml','--include=*.log','--include=*.csv','--include=COMPLETE','--exclude=*','turing:/scratch/node07/aryama.murthy/family-eval128-20260908/',str(artifacts)+'/'],check=True,timeout=90)
  subprocess.run([sys.executable,'scripts/summarize_family_eval128.py','--root',str(artifacts),'--report',str(report)],check=True,stdout=subprocess.DEVNULL,timeout=30)
  status=json.loads((report/'status.json').read_text())
  queue=subprocess.check_output(['ssh','turing','squeue -h -j 28982 -o "%i %T %M %R"'],text=True,timeout=30)
  (report/'queue.txt').write_text(queue)
  print(time.strftime('%Y-%m-%d %H:%M:%S'),status['complete_lanes'],'/64 complete',flush=True)
  if status['complete'] or not queue.strip():
   accounting=subprocess.check_output(['ssh','turing','sacct -j 28982 --format=JobID,State,Elapsed,ExitCode,AllocTRES -P'],text=True,timeout=30)
   (report/'accounting.txt').write_text(accounting)
   lines=['# 128-question evaluation, 2,048-token cap','', 'Status: '+('COMPLETE' if status['complete'] else 'INCOMPLETE — inspect accounting and lane logs.'),'','| Pair | Mapper | Requests | Exact vs FP32 AR | Decode tokens/s | Request tokens/s |','|---|---|---:|---:|---:|---:|']
   for family,methods in status['families'].items():
    for method,r in methods.items():
     lines.append(f"| {family} | {method} | {r['requests']} | {r['exact_matches']}/{r['paired_requests']} | {r['decode_tps'] or 0:.2f} | {r['request_tps'] or 0:.2f} |")
   lines.extend(['','Full sequence comparisons and first-divergence positions are in status.json. Raw token IDs, timing, and hashes are in artifacts/. See PLAN.md and protocol.json for selection and precision. These are token-sequence matches, not task-answer accuracy.'])
   (report/'RESULTS.md').write_text('\n'.join(lines)+'\n')
   break
 except Exception as exc:
  print(type(exc).__name__,str(exc),flush=True)
 time.sleep(45)
