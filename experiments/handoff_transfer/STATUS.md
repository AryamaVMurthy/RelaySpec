# Handoff transfer status

This campaign follows the exact handoff.zip source, not the earlier AUF
five-map or rank32 drafter-LoRA recipes.

- Archive SHA25650ef7120aee5b9087356814eee4e235281437a6fa99ba5097ff5d86932002d45.
  All37 hashes and the unmodified packaged CPU loss check pass.
- Pinned SpecForge953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef checked out and
  copied to the remote control directory. Vendor source is unchanged.
- Original Qwen3-4B target/drafter revisions exactly match the handoff and
  were reused. The pinned Math target adapter was downloaded separately.
- CPU preparation31522 COMPLETED exit0. PEFT0.20.0 and accelerate1.12.0
  installed in an isolated overlay; existing shared runtime not changed.
  Torch/vLLM/Transformers match the handoff; pinned SpecForge imports.
- The author's original feature cache path is permission-denied. No access
  workaround attempted; our own fidelity trajectories/features will be generated.
- Gate31523: pending AssocGrpGRES, requests2 GPUs on node07 after31450.
  Runs32 generated responses, dense capture, exact packaged2-update DDP fit,
  frozen-weight/export checks, then4 AR/native/mapper exactness checks.
  First-four eval checks are preliminary and do not by themselves establish
  comprehensive long/EOS/length-stop coverage; add representative cases
  before declaring the full fidelity gate complete.
- Full origin31525 follows31523:4096 records,2000 updates, then128x2048
  AR/native/mapper comparisons against bundled references.31524 was replaced
  before execution to include the full-checkpoint frozen/export verification.
- No new GPU handoff result is available yet. Transfer ports, normal-RelaySpec
  matched fitting and three-way transfer evaluations still need implementation.
- Held all21 earlier pending arrays (31503,31502,31500,31498,31496,31495,
  31494,31492,31491,31490,31489,31488,31487,31486,31481,31480,31479,31478,
  31472,31471,31470). Their outputs and definitions remain intact. Existing
  live jobs were allowed to finish; old results retain their recipe labels.

Plan: docs/plans/2026-09-11-handoff-transfer.md. Shared maximum remains4 GPUs;
external account quota can delay allocation even when fewer study GPUs run.
