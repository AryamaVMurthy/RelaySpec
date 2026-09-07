import unittest,sys,json,ast,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from sampling import check,positions
class PackageTests(unittest.TestCase):
 def test_sampling(self):self.assertTrue(check()['passed'])
 def test_all_reference_tokens(self):
  sets=[]
  for side in ['ar8','native8','mapped']:
   rows=[json.loads(l) for l in (R/'reference'/f'{side}.jsonl').read_text().splitlines()]
   self.assertEqual(len(rows),128);self.assertEqual(len({r['group_id'] for r in rows}),128)
   self.assertEqual(sum(len(r['output_ids']) for r in rows),124899);sets.append(rows)
  for trio in zip(*sets):
   self.assertEqual(len({tuple(r['output_ids']) for r in trio}),1)
   self.assertEqual(len({tuple(r['prompt_ids']) for r in trio}),1)
 def test_sources_parse(self):
  for p in R.rglob('*.py'):
   if not any(x in p.parts for x in ['work','.venv']):ast.parse(p.read_text(),filename=str(p))
 def test_partition(self):
  for workers in [2,4]:self.assertEqual(sorted(sum([list(range(128))[i::workers] for i in range(workers)],[])),list(range(128)))
 def test_fold_and_objective(self):
  import torch
  from mapper import Context,relative,rms
  torch.manual_seed(42);f=torch.randn(3,15);norm=torch.randn(3);m=Context(f,norm,d8=4,d4=3,layers=5)
  x=torch.randn(7,20);y=torch.randn(7,15)
  c,z=m(x);folded=m.frozen_norm(torch.nn.functional.linear(x,m.folded()))
  torch.testing.assert_close(c,folded,rtol=2e-5,atol=2e-5)
  manual=relative(c,rms(torch.nn.functional.linear(y,f))*norm)
  manual+=sum(relative(z[:,i*3:(i+1)*3],y[:,i*3:(i+1)*3]) for i in range(5))/5
  torch.testing.assert_close(m.loss(x,y),manual)
  m.loss(x,y).mean().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_bundle_has_no_model_artifacts(self):
  bad={'.safetensors','.pt','.pth','.bin','.gguf','.whl','.parquet','.onnx'}
  self.assertFalse([str(p) for p in R.rglob('*') if p.is_file() and p.suffix in bad and 'work' not in p.parts and '.venv' not in p.parts])
if __name__=='__main__':unittest.main()
