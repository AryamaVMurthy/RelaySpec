import pathlib,shutil,json,hashlib
root=pathlib.Path('/scratch/aryama.murthy/transfer-reproduction-20260907')
p=root/'quick512-package'; w=root/'quick512-work'
assert not p.exists() and not w.exists()
shutil.copytree(root/'package',p,ignore=shutil.ignore_patterns('.venv','__pycache__','work'))
(p/'.venv').symlink_to(root/'package/.venv',target_is_directory=True)
w.mkdir()
for name in ['models','evaluation']:(w/name).symlink_to(root/'work'/name,target_is_directory=True)
(w/'common/rollouts/train').mkdir(parents=True)
for i in range(4):
 for ext in ['jsonl','meta.json']:
  name=f'{i:05d}.{ext}';(w/'common/rollouts/train'/name).symlink_to(root/'work/common/rollouts/train'/name)
changes={}
for name in ['train.py','capture.py','check_features.py']:
 f=p/'src'/name;s=f.read_text();before=hashlib.sha256(s.encode()).hexdigest();s=s.replace('16384','512')
 if name=='check_features.py':s=s.replace('len(files)==128','len(files)==4')
 f.write_text(s);changes[name]={'before':before,'after':hashlib.sha256(s.encode()).hexdigest()}
(w/'tmp').mkdir()
(w/'pilot_contract.json').write_text(json.dumps({'training_examples':512,'epochs':3,'evaluation_count':8,'output_cap':512,'historical_reproduction':False,'source':'First four complete freshly generated shards from28637; distinct evaluation prompts','changes':changes},indent=2))
