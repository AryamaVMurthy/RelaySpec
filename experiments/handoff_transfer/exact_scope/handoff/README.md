# Yashas handoff — 2026-09-12

All experiments were stopped at the user's request. Jobs 32633 (long pilots),
32637 (full matched table) and 32638 (collector) were cancelled. Do not resume
automatically. Completed exactness evidence is eight Qwen requests at cap512:
`invariant-smalltile-rms` achieved 8/8 identical AR/native token sequences,
41.10 AR TPS and 196.48 native TPS. The cap2048 pilots did not finish and the
128-request optimized table did not run. See `../EXACTNESS_SEARCH.md`.

## GitHub

Repository: https://github.com/AryamaVMurthy/RelaySpec

- `research/auf-vllm-20260911`: current CE/AUF, transfer, matched evaluation,
  numerical diagnosis code, configuration and audited reports.
- `main`: paper, native-method experiments, hardware profiling and historical
  scaling/fitting records.
- `research/scaling-autoresearch-20260905`: scaling, competitor and cross-family
  studies and their measurement records.

Large model weights and feature caches remain in shared cluster scratch.
Duplicate compressed source archives and local process-ID files are not added
to Git. Source, configurations, paper assets and text measurement records are
versioned. The supplied ZIP contents are implemented in the relevant transfer
experiments; original ZIP archives remain in the local workspace.

## Turing access

Named user: `yashas.kotre`, UID `2024101028`.
Read/write file access and directory traversal were granted recursively on
node07, preserving executable-file permissions. Directory default ACLs also
grant this user access to normally created future files. Symlinks are not
followed when changing permissions. The verifier reads each changed ACL back.
See `acl-verification.json` for the verified scope/count.

On node07:

```
/scratch/aryama.murthy/
/scratch/aryama.murthy/handoff-transfer-20260911/
/scratch/aryama.murthy/relayspec-auf-20260911/
/scratch/aryama.murthy/transfer-reproduction-20260907/
/scratch/aryama.murthy/yashas-handoff-20260912/control/
```

On the login node the same scratch paths start with
`/scratch/node07/aryama.murthy/`. Named ACLs are applied/verified locally on the
compute node: this NFS client does not expose them through `getfacl`.

The home filesystem is ZFS mounted `noacl`. A current copy of the control
source, logs and outputs is provided under the shared `control/` path above;
private home permissions are unchanged. No password, SSH key or login identity
is shared.

A scratch-hosted Python base and virtual-environment entry point avoid the
original environment's interpreter symlink into private home:

```
/scratch/aryama.murthy/yashas-handoff-20260912/venv/bin/python
```

Its site-packages refer to the existing, now-shared vLLM environment. CPU import
of torch and vLLM was checked on node07. Use your own Turing login and Slurm
allocation/account to run GPU work. File access does not grant another user's
Slurm account privileges. Existing Slurm scripts still contain Aryama's home
paths/account: copy them and update their control, log and account paths before
submitting. Do not run inference on the login node.

Example environment inside an allocated node07 GPU job:

```bash
cd /scratch/aryama.murthy/yashas-handoff-20260912/control
export PYTHONPATH="$PWD/experiments/auf_vllm/runtime_zip:$PWD/src:$PWD"
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_BATCH_INVARIANT=1 RELAYSPEC_NUMERICS=invariant-smalltile-rms
python_path=/scratch/aryama.murthy/yashas-handoff-20260912/venv/bin/python
# Use "$python_path" to invoke the selected experiment module.
```

Current results: `experiments/handoff_transfer/exact_scope/reports/` in the
AUF branch, control `outputs/`, and raw scratch campaign directories. Historical
complete invariant-runtime comparisons and optimized AR-only comparisons are
separate from the unfinished near-peak exactness confirmation.
