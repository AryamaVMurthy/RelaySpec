"""vLLM acceptance counters: exclude warmup/retries and aggregate by steps."""
FIELDS={'vllm:spec_decode_num_drafts':'verification_iterations','vllm:spec_decode_num_accepted_tokens':'accepted_draft_tokens','vllm:spec_decode_num_draft_tokens':'proposed_draft_tokens'}
def snapshot(engine):
 result={}
 for metric in engine.get_metrics():
  name=metric.name if hasattr(metric,'name') else metric['name']
  if name in FIELDS:
   value=metric.value if hasattr(metric,'value') else metric['value']
   result[FIELDS[name]]=result.get(FIELDS[name],0)+int(value)
 assert set(result)==set(FIELDS.values()),'Enable vLLM statistics; speculative counters are missing'
 return result
def delta(before,after):
 result={k:after[k]-before[k] for k in FIELDS.values()}
 assert all(v>=0 for v in result.values())
 assert result['accepted_draft_tokens']<=result['proposed_draft_tokens']
 return result
def aggregate(rows):
 if not rows or any('verification_iterations' not in x or 'accepted_draft_tokens' not in x for x in rows):return None
 n=sum(x['verification_iterations'] for x in rows);a=sum(x['accepted_draft_tokens'] for x in rows)
 if not n:return None
 assert n>0 and a>=0
 result={'verification_iterations':n,'accepted_draft_tokens':a,'mean_acceptance_length':1+a/n,'mean_accepted_draft_tokens':a/n,'definition':'1 + total accepted draft tokens / total speculative verification steps; vLLM bonus-token convention; step-weighted, not mean of prompt means.'}
 if all('proposed_draft_tokens' in x for x in rows):
  p=sum(x['proposed_draft_tokens'] for x in rows)
  result.update(proposed_draft_tokens=p,draft_acceptance_rate=a/p if p else None)
 return result
