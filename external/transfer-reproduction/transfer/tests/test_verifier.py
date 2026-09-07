"""Exercise the real command on reference output and deliberate corruptions."""
import unittest,tempfile,subprocess,os,sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
class VerifierTests(unittest.TestCase):
 def run_case(self,corrupt=None):
  with tempfile.TemporaryDirectory(dir=R/'tests') as tmp:
   root=Path(tmp)
   for side in ['ar8','native8','mapped']:
    rows=[json.loads(l) for l in (R/'reference'/f'{side}.jsonl').read_text().splitlines()]
    for x in rows:x.update(timing_valid=True,output_tokens=len(x['output_ids']))
    if corrupt=='one' and side=='mapped':rows[0]['output_ids'][0]+=1
    if corrupt=='all':rows[0]['output_ids'][0]+=1
    if corrupt=='duplicate' and side=='mapped':rows[1]=rows[0]
    p=root/'measurements/final/worker_0';p.mkdir(parents=True,exist_ok=True)
    (p/f'{side}-r0.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
   e=os.environ.copy();e['TRANSFER_WORK']=tmp;e['PYTHONDONTWRITEBYTECODE']='1'
   q=subprocess.run([sys.executable,'-B',str(R/'src/verify.py')],env=e,capture_output=True,text=True)
   return q.returncode
 def test_accepts_full_reference(self):self.assertEqual(self.run_case(),0)
 def test_rejects_one_pipeline_difference(self):self.assertNotEqual(self.run_case('one'),0)
 def test_rejects_common_drift_from_reference(self):self.assertNotEqual(self.run_case('all'),0)
 def test_rejects_duplicate_coverage(self):self.assertNotEqual(self.run_case('duplicate'),0)
if __name__=='__main__':unittest.main()
