# Locked experiment configuration — batch8 per GPU

Selected node: Turing node07, four L40S GPUs. Each token fit uses exactly one GPU
and batch8, without gradient accumulation or cross-GPU gradient synchronization.
Up to four different jobs run concurrently. This supersedes the two-GPU lanes.
No further training batch-size tuning. Batched warmup is used in subsequent vLLM jobs; warmup and any compilation-affected first pass remain outside timing. Dataset and evaluation selections stay fixed.

## Transfers

| Family | Source/drafter interface | Frozen new target |
|---|---|---|
| q8 | Qwen3-4B | Qwen3-8B |
| llama | Llama3.1-8B | Llama3.2-3B |
| cross | Qwen3-4B | Llama3.1-8B |

Qwen uses a shared tokenizer/token-ID vocabulary. Cross-family runs use the explicit
text/tokenizer alignment bridge; they are not evidence of direct vocabulary-ID reuse.
All target transformers and all drafter parameters outside the named interfaces remain
frozen. There is no target LoRA. Existing Qwen/Llama initializer checkpoints are reused;
new cross-family original/ZIP initializers are fit for one epoch each.

## Fits per family

| Cell | Trainable interface | Objective |
|---|---|---|
| normal_ce/ce | Original RelaySpec normalized linear interface | Token CE |
| dense_fusion/ce | ZIP initializer folded into one fusion matrix | Token CE |
| five_maps/ce | Five dense matrices, frozen native fusion | Token CE |
| five_maps/auf | Five dense matrices, frozen native fusion | AUF |
| dense_fusion/auf | One full fusion matrix | AUF |
| five_ba56/ce | Five BA updates, rank56, alpha56 | Token CE |
| five_ba56/auf | Five BA updates, rank56, alpha56 | AUF |
| feature_ce | Original normalized linear interface | Soft-target feature CE |
| forward_kl | Original normalized linear interface | KL(source || mapped) |
| reverse_kl | Original normalized linear interface | KL(mapped || source) |

ZIP+CE and one-fusion-matrix+CE are one cell, not duplicate experiments.
Thirty comparison fits: ten per family. Baseline evaluations per family: AR,
unchanged original RelaySpec feature-MSE initializer, unchanged ZIP initializer.
AUF supervises the correct prefix plus its first failure. No exponential positional
weighting or additional MSE/KL term is mixed into token CE/AUF. Feature CE/KL uses
softmax over normalized fused feature coordinates at temperature1; no extra positional
weights. Feature CE and forward KL have identical gradients with a fixed teacher.

## Token fitting

- 4096 distinct training records; rollout cap4096 generated tokens.
- One epoch: 512 updates ×8 records =4096 record presentations.
- Up to32 distinct eligible anchors per record, sampled in the response; short records
  provide fewer. Clean anchor and padding excluded from prediction loss.
- Per GPU batch8; one GPU per fit; accumulation1. CE/AUF normalized within this batch8.
- BF16 operations, FP32 trainable interface weights/loss reductions, cached frozen features.
- Fused AdamW, weight decay0, gradient clip1, 5% warmup then cosine decay; seed42.
- Loss-processing chunks32 blocks; native checkpoint block sizes retained.
- Final checkpoint only for main comparisons. A two-update check verifies fit/export
  integrity before each full same-family fit; full128 evaluation verifies deployment.
- Different batching changes random anchor draws and floating-point reductions relative
  to the archived two-GPU pilot; all new comparison cells use the same batch8 protocol.

| Interface | Qwen4→8 LR | Llama8→3 LR | Qwen4→Llama8 LR |
|---|---:|---:|---:|
| Original / five dense maps | 1e-4 | 1e-4 | 1e-4 |
| Dense fusion | 1e-4 | 6e-4 | 1e-4 |
| Five BA56 | 3e-4 | 6e-4 | 1e-4 |

Rates reuse existing validation selection; cross-family rates are untuned. No fresh LR,
seed, capacity, data-size or regularization sweep. Feature-only fits use one cache epoch,
position batch2048 and LR1e-3; these bypass block anchors and are not512-update token fits.

## Evaluation

- vLLM0.28 primary backend; same fixed128 requests per family, cap2048 output tokens.
- Greedy temperature0, natural EOS, seed0; cap is a maximum, not forced length.
- Submit128 requests together; max_num_seqs128, max_num_batched_tokens8192,
  max_model_len5120, GPU memory utilization0.8, prefix cache disabled.
- vLLM schedules active requests within KV-cache capacity; submission batch128 does not
  assert all128 remain resident simultaneously. This is a target setting, not a measured
  maximum until the full run succeeds. Only explicit memory failures trigger128→64→32→16→8.
- Lock the passing serving batch for AR, original, ZIP and all methods within a family.
- Three timing repetitions, one training seed. Setup/compilation/warmup excluded; logged.
- Report aggregate generated tokens / batch wall time, speedup against matched AR and
  original/ZIP, acceptance/progress, full output-token equality and finish equality.
- GPU utilization, memory, power and temperature recorded. Batched throughput is distinct
  from the older single-request latency numbers.
- Final Transformers checks remain limited to1–2 representative completed cases.

## Initial parallel launch

Jobs32239–32242 on node07: Qwen five_maps/CE, five_maps/AUF, dense_fusion/CE,
and five_ba56/AUF respectively. Each requests one GPU and executes batch8.
Llama token/paired caches are being staged from node06, which was fully allocated.
All remaining fits, reference evaluations and cross-family data preparation are
submitted under the same total four-GPU ceiling. Each family first completes its shared baselines and serving-batch selection;
remaining evaluations then run independently, reading the completed reference
artifacts. Reference comparison JSON writes are atomic. Superseded job ledgers/results are retained separately.

## Execution evidence
Initial four batch8 fits completed with export verification in285.22–292.44 fitting
seconds each. Qwen batch128 AR repetition0 completed128 requests,122768 generated
tokens,99.8542 measured seconds,1229.4725 aggregate tokens/s. No trained-method
speedup claim follows until its matching full output comparison passes.
Transformers jobs32304–32306 run one128-request/2048-cap pass each for AR and Qwen
five-map CE/AUF after the entire vLLM matrix. The standalone decoder currently
supports one request at a time; these are backend confirmation checks with separately
reported timing, not substitutes for the batch128 primary results.

After the final audit passes, its CPU job archives compact deployment interfaces,
unmerged token-fit parameters, model/config provenance, calibration manifests,
evaluation token outputs and available GPU telemetry under the control directory's
`outputs/exact32e1b8/artifacts`. The archive has a SHA256 manifest; large frozen
model weights and feature caches remain referenced rather than duplicated.
The collector also produces CSV/Markdown tables from complete audited timing
triplets, preserving benchmark job IDs and physical GPU UUIDs in the audit.
It checks exact group-ID disjointness between each family's fitting records,
evaluation requests and warmup requests, using the assembled index for cross-family
data. This is an exact-ID check, not a semantic near-duplicate assessment.
