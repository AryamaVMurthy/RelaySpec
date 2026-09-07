from pathlib import Path
import json,re,hashlib,unicodedata,time,collections
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from paths import WORK,PACKAGE,model
R=WORK
(R/"common/dataset").mkdir(parents=True,exist_ok=True)
(R/"validation").mkdir(parents=True,exist_ok=True)
def normalize(s):
 if not isinstance(s,str) or not s.strip():return None
 return " ".join(unicodedata.normalize("NFKC",s).lower().split())
def groupkey(s):
 s=normalize(s)
 if s is None:return None
 # Conservative exact/template grouping; this is custom curation, not official.
 s=re.sub(r"\d+(?:\.\d+)?","#",s)
 s=re.sub(r"[^\w#]+"," ",s)
 return hashlib.sha256(" ".join(s.split()).encode()).hexdigest()
assert normalize(None) is None and normalize("") is None
assert normalize(" A\n B ")==normalize("a b")
assert groupkey("Compute 13 + 4.")==groupkey("Compute 17 + 8.")
assert groupkey("Compute 13 + 4.")!=groupkey("Factor x squared.")
assert normalize("ＡＢＣ")==normalize("abc")
t=time.perf_counter()
tok=AutoTokenizer.from_pretrained(model(8))
tok4=AutoTokenizer.from_pretrained(model(4))
assert tok.get_vocab()==tok4.get_vocab()
for key in ["bos_token_id","eos_token_id","pad_token_id"]:
 assert getattr(tok,key)==getattr(tok4,key)
rows=pq.read_table(WORK/"dataset/train-00000.parquet",columns=["problem","source"]).to_pylist()
seen=set();groups=[];counts=collections.Counter()
for idx,row in enumerate(rows):
 k=groupkey(row["problem"])
 if k is None:counts["malformed"]+=1;continue
 if k in seen:counts["duplicate_or_template"]+=1;continue
 seen.add(k)
 row.update(row_id=idx,group_id=k)
 groups.append(row)
# Deterministic representative sample from each source, then exact proportional quotas.
strata=collections.defaultdict(list)
for row in groups:strata[row["source"]].append(row)
for source,items in strata.items():
 items.sort(key=lambda x:hashlib.sha256(("42:"+x["group_id"]).encode()).digest())
total=16384+256+4096
raw={s:total*len(v)/len(groups) for s,v in strata.items()}
quotas={s:int(v) for s,v in raw.items()}
for s in sorted(raw,key=lambda s:raw[s]-quotas[s],reverse=True)[:total-sum(quotas.values())]:
 quotas[s]+=1
selected=[]
for source,items in sorted(strata.items()):
 accepted=0
 for row in items:
  user=row["problem"]+"\nSolve the problem and put your final answer within \\boxed{}."
  prompt=tok.apply_chat_template([{"role":"user","content":user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=tok.encode(prompt,add_special_tokens=False)
  if len(ids)>1024:counts["prompt_over_1024"]+=1;continue
  assert len(ids)>0
  row.update(prompt_token_ids=ids,prompt=prompt)
  selected.append(row);accepted+=1
  if accepted==quotas[source]:break
 assert accepted==quotas[source],source
# Partition each stratum proportionally; exact global sizes via stable global shuffle.
selected.sort(key=lambda x:hashlib.sha256(("split42:"+x["group_id"]).encode()).digest())
parts={"train":selected[:16384],"dev":selected[16384:16640],"eval":selected[16640:]}
assert [len(parts[k]) for k in ["train","dev","eval"]]==[16384,256,4096]
assert len({x["group_id"] for xs in parts.values() for x in xs})==total
for name,items in parts.items():
 p=R/"common/dataset"/(name+".jsonl")
 assert not p.exists()
 with p.open("w") as f:
  for x in items:f.write(json.dumps(x,ensure_ascii=False)+"\n")
summary={"source_scope":"first pinned upstream shard; not full corpus","revision":"9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
"rows_scanned":len(rows),"unique_template_groups":len(groups),"filters":dict(counts),
"dedup_contract":"NFKC lowercase whitespace exact and number/punctuation normalized template groups; no semantic equivalence claim",
"partition_source_counts":{k:dict(collections.Counter(x["source"] for x in v)) for k,v in parts.items()},
"counts":{k:len(v) for k,v in parts.items()},"wall_seconds":time.perf_counter()-t,
"tokenizer_vocab_equal":True,"training_solutions_used":False,
"hashes":{k:hashlib.sha256((R/"common/dataset"/(k+".jsonl")).read_bytes()).hexdigest() for k in parts}}
(R/"common/dataset/selection.json").write_text(json.dumps(summary,indent=2))

# Evaluation uses the published math suffix, distinct from training generation.
for split,items in [('eval',parts['eval'][:128]),('warmup',parts['dev'])]:
 expected=[json.loads(l) for l in (PACKAGE/'reference'/('eval_prompts.jsonl' if split=='eval' else 'warmup_prompts.jsonl')).read_text().splitlines()]
 actual=[]
 for row,ref in zip(items,expected):
  row=dict(row);user=row['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'
  row['prompt']=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  row['prompt_token_ids']=tok.encode(row['prompt'],add_special_tokens=False)
  assert row['group_id']==ref['group_id'] and row['prompt_token_ids']==ref['prompt_token_ids'], 'Prompt reconstruction differs'
  actual.append(row)
 p=WORK/'evaluation';p.mkdir(exist_ok=True)
 (p/(split+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in actual))
expected=json.loads((PACKAGE/'provenance/selection.json').read_text())
assert summary['hashes']==expected['hashes'], 'Training selection differs from reference'
print('Dataset and all reference prompts reconstructed exactly')
