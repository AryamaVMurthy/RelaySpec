"""Resume only a matching, atomically saved evaluation prefix."""
import json,math,shutil
def restored(dst,data,args):
 if not dst.exists():return []
 d=json.loads(dst.read_text())
 expected={'backend':'vllm','compilation_mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','batch_invariant':True,'async_scheduling':False,'cap':args.cap,'worker':args.worker,'workers':args.workers,'side':args.side}
 assert all(d.get(k)==v for k,v in expected.items()),'Saved evaluation protocol differs'
 rows=d['rows'];assert len(rows)<=len(data)
 assert [x['group_id'] for x in rows]==[x['group_id'] for x in data[:len(rows)]], 'Saved prompt ownership/order differs'
 assert len({x['group_id'] for x in rows})==len(rows)
 assert all(x['output_ids'] and math.isfinite(x['seconds']) and x['seconds']>0 for x in rows)
 backup=dst.parent.parent/'resume_archive'/dst.parent.name/dst.name
 backup.parent.mkdir(parents=True,exist_ok=True)
 if not backup.exists():shutil.copy2(dst,backup)
 return rows
