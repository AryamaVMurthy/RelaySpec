"""Fixed 25% stratified sampling; prompt and response have independent strata."""
import hashlib,json,random

def key(ids):
 return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()

def positions(n,boundary,group):
 assert 0<=boundary<=n
 out=[]
 for tag,start,end in [('prompt',0,boundary),('response',boundary,n)]:
  seed=int.from_bytes(hashlib.sha256(('42:'+str(group)+':'+tag).encode()).digest()[:8],'little')
  rng=random.Random(seed)
  for lo in range(start,end,4):out.append(lo+rng.randrange(min(4,end-lo)))
 return out

def check():
 for n in range(70):
  for b in range(n+1):
   p=positions(n,b,'test');assert p==sorted(set(p));assert len(p)==(b+3)//4+(n-b+3)//4
   for start,end in [(0,b),(b,n)]:
    for lo in range(start,end,4):assert sum(lo<=x<min(lo+4,end) for x in p)==1
   assert p==positions(n,b,'test')
 for bad in [-1,11]:
  try:positions(10,bad,'x')
  except AssertionError:pass
  else:raise AssertionError('invalid boundary accepted')
 return {'passed':True,'cases':'all boundaries at lengths 0..69; partial strata; invalid boundaries; determinism'}
if __name__=='__main__':print(json.dumps(check()))
