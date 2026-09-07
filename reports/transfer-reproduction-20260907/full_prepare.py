from pathlib import Path
import shutil,json,hashlib
r=Path('/scratch/aryama.murthy/transfer-reproduction-20260907');p=r/'full-package';w=r/'work'
assert not p.exists()
shutil.copytree(r/'package',p,ignore=shutil.ignore_patterns('.venv','__pycache__','work'))
(p/'.venv').symlink_to(r/'package/.venv',target_is_directory=True)
f=p/'src/capture.py';s=f.read_text().replace("llm=LLM(model=str(model(a.size)),runner=", "llm=LLM(model=str(model(a.size)),enforce_eager=True,runner=");f.write_text(s)
f=p/'reproduce.py';s=f.read_text().replace("if profile=='capture':e['TRANSFER_CAPTURE']='1'", "if profile=='capture':e['TRANSFER_CAPTURE']='1';e['VLLM_BATCH_INVARIANT']='1'");f.write_text(s)
for name in ['mapper.py','train.py','export.py']:
 assert (p/'src'/name).read_bytes()==(r/'package/src'/name).read_bytes()
# Reuse only exact complete paired pilot feature shards, same full token sequences and capture configuration.
for size in ['4','8']:
 dst=w/'cache/features/train'/size;dst.mkdir(parents=True)
 for i in range(4):
  for ext in ['pt','meta.json']:
   name=f'{i:05d}.{ext}';(dst/name).symlink_to(Path('/scratch/aryama.murthy/rs-q512/cache/features/train')/size/name)
contract={'training_examples':16384,'epochs':3,'fresh_training_realization':True,'historical_rollout_hash_match':False,'capture':'eager and batch invariant, strict1e-6 sampled/dense check unchanged','loss_train_export_identical_to_archive':True,'reused_feature_shards_per_model':4,'reason':'All pinned models/software/dataset/prompts and two-worker queue match; current driver differs. Preserve original data and explicitly reproduce recipe on fresh rollouts; do not claim bitwise historical reproduction.'}
(w/'full_run_contract.json').write_text(json.dumps(contract,indent=2))
