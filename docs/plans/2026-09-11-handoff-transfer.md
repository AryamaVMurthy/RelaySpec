# Exact Handoff Recipe and Three-Way Transfer Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce handoff.zip's actual AUF fusion adaptation, transfer it across the paper's models, and compare it with the older ZIP recipe and normal RelaySpec on matched128-request,2048-token evaluations.

**Architecture:** Preserve the packaged objective and pinned SpecForge block implementation. Separate an unchanged same-model fidelity check from transfer ports that require different feature dimensions, initial fusion matrices, or token interfaces. Keep all original targets frozen in the transfer study; only the fidelity check uses the supplied frozen Math target adapter.

**Tech Stack:** Torch2.13.0+cu129, vLLM0.28.0+cu129, Transformers5.16.1, PEFT0.20.0; SpecForge953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef; max4 L40S GPUs; isolated dependency overlay and Slurm jobs.

## 1. Verified source and changes from the previous study

Archive SHA256:50ef7120aee5b9087356814eee4e235281437a6fa99ba5097ff5d86932002d45.
All37 bundled file hashes verified. Packaged check_loss.py passes unchanged.
Immutable source is experiments/handoff_transfer/vendor/code; data is in
data/handoff-20260911/handoff/data. Do not edit the vendor files.
The upstream repository is checked out at its exact commit in data/specforge-handoff.

The handoff trains fusion LoRA only, not drafter-body LoRA. Its gains are
against native speculative decoding of a LoRA-adapted target, not against
normal RelaySpec. Earlier AUF mapper or rank32 continuation measurements do
not reproduce this recipe and must retain their old names in reports.

| Setting | Exact handoff contract |
|---|---|
| Trainable module | fc only; PEFT rank=alpha=56; dropout0; no bias |
| Initialization | Native frozen F0; seeded PEFT A; B=0 |
| Loss | Packaged objectives.py AUF; first wrong valid proposal included |
| Frozen | Target, drafter body, native embedding/head, original fc, RMSNorm |
| Data | 4096 unique target-generated responses; response cap4096; natural EOS |
| Features | Dense BF16 target states at the same saved token prefixes |
| Anchors | Up to8 distinct response anchors per presentation; SpecForge blocks/mask |
| Updates | Exactly2000,16000 total record presentations |
| Batch | 2 GPUs; microbatch2/GPU; accumulation2; effective8 records/update |
| Reduction | Chunk numerators/denominators summed within microbatch; equal microbatch means across accumulation/DDP |
| Optimizer | AdamW1e-4; betas0.9/0.999; eps1e-8; decay0; clip1 |
| Schedule | Packaged step calculation;100-step warmup and cosine over2000 steps |
| RNG | Exact shard shuffle, length-bin sort, rank assignment and anchor seeds from train.py |
| Checkpoint | Fixed final endpoint; no learning-rate/checkpoint selection |
| Export | FP32 F0+BA, castBF16; no low-rank runtime overhead |

Preserve source arithmetic, not merely a prose approximation. In particular,
the package multiplies warmup and cosine terms before optimizer.step.

## 2. Experiment matrix and fair comparisons

### Fidelity check

Qwen3-4B + the pinned Math target adapter, original Qwen3-4B DFlash drafter.
Use exact bundled Math train/eval manifests. First32 records/two updates
are a diagnostic gate, then full4096/2000 and128x2048 reproduction.
Only this arm loads a target LoRA, as required to verify the supplied recipe.
Other original domains are not prerequisites for the requested transfer comparison.
The package's KiCad8192 cap is not imported into the primary2048-cap study.

### Priority transfer cases

| Case | Source drafter -> frozen target | Handoff fusion dimensions | Trainable rank56 factors |
|---|---|---|---:|
| Q8 | Qwen3-4B -> Qwen3-8B | 2560 x20480 | 1,290,240 |
| Q14 | Qwen3-4B -> Qwen3-14B | 2560 x25600 | 1,576,960 |
| L3 | Llama-3.1-8B -> Llama-3.2-3B | 4096 x15360 | 1,089,536 |
| X8 | Qwen3-4B -> Llama-3.1-8B | Dimension/vocabulary audit required | Conditional on valid interface |

For each supported case evaluate:
1. Normal RelaySpec: original normalized-input direct context mapper and
   relative context loss from src/relayspec/relay.py and losses.py.
2. Older ZIP: five linear layer maps, layer+normalized-context feature loss.
3. Handoff transfer: freeze the corresponding older-ZIP folded F0; train only
   rank56 PEFT fusion residual with the exact handoff loss/sampling/schedule.
4. Same-runtime AR, and native target drafter where a compatible released model exists.

A native F0 cannot be copied unchanged across hidden widths. Therefore item3
is explicitly a transfer extension with ZIP initialization, not a claim of
unchanged native initialization. Charge ZIP's fitting/source-feature costs
and three initial epochs in addition to the2000 handoff updates. All three
methods see the same4096 unique target trajectories. Reuse existing ZIP
checkpoints only after data/model/source/export provenance matches.

Do not call the recipe comparison a loss-only or equal-compute comparison.
Add one Q8 control: identical handoff fusion/rank/data/update recipe with
uniform CE, isolating AUF support weighting. Also report matched measured
GPU-time adaptation comparisons on Q8, charging initialization. No broad
learning-rate/epoch/MLP sweep precedes the principal result.

## 3. Evaluation contract

- First completed table: same128 Numina development prompts per supported
  target, maximum2048 generated tokens, natural EOS; original frozen target.
- Then frozen MATH/GSM8K/code/chat development manifests already prepared
  for this study,128 prompts per workload. Keep untouched confirmation
  manifests reserved until all choices are frozen.
- Greedy temperature0,top_p1,top_k disabled, seed0, thinking off for Qwen.
  Single-request sequential latency; vLLM batching is for generation/capture,
  not silently added to the single-request inference benchmark.
- One common vLLM configuration per target across all arms; same prompt IDs,
  verifier, context limit, block length, precision and cache policy. Qwen
  block16; Llama's released block/token-mask conventions must be audited.
  Never copy Qwen mask token151669 into the Llama vocabulary.
- Three timing repetitions for final checkpoints; rotate arm/GPU assignment.
  One training seed42 initially. Add43/44 only to the Q8/L3 main comparison
  after correctness, without multiplying exploratory fits.
- Warmup/setup and first-use compilation times are recorded separately.
  Report TPS=sum(output_tokens)/sum(request_wall_seconds), ratios of aggregate
  throughputs, exact full-output and finish-reason matches, acceptance length,
  draft-only acceptance, accepted/proposed fraction, position survival,
  latency and training GPU-hours. Profile separate runs, not timed ones.
- Any full-output disagreement triggers diagnosis. Fresh AR references may
  replace incompatible bundled references only after documenting the cause;
  do not relax equality or alter the verifier to obtain a pass.

## 4. Cross-family engineering gate

Audit every source/target tokenizer, ID vocabulary, mask/EOS conventions,
embedding/head width and proposal/output mapping. Qwen and Llama IDs are not
interchangeable. The handoff itself contains no heterogeneous-vocabulary bridge.
First test the existing historical greedy text bridge on representative
short/long/EOS examples, then audit source-token supervision against target
rollout character-prefix boundaries and SpecForge attention positions.
Source-token AUF and target-token acceptance must be distinguished.
If a valid bridge requires changing the loss support, training additional
embedding/head parameters or a different verifier, record it as a separate
extension rather than an exact handoff port. Full128x2048 X8 training/timing
must follow a passing correctness gate; do not fill its result with unrelated
historical numbers. Continue supported cases while engineering this gate.

## 5. Tasks and execution order

### Task1: Immutable package and prerequisites
Files: vendor/, reports/package-audit.json, prepare_origin.py/.sbatch.
Verify SHA256SUMS; run vendor/code/check_loss.py; obtain pinned SpecForge;
stage the exact existing Qwen base/draft revisions and pinned Math adapter.
Install missing PEFT in a separate overlay, never mutate the shared live env.
Expected: prepared.json with exact paths/versions and a successful upstream import.
The supplied author's cache locator is permission-denied under this account;
regenerate our own gate data, without attempting access workarounds.

### Task2: Golden GPU gate and full origin check
Create origin_gate.sbatch and gate verification utility. Allocate2 GPUs,
generate32 responses in a bulk batch, capture dense features, inspect
token-prefix identity and causality, run packaged2-update DDP training.
Check only fc A/B changed, finite gradients/loss, expected860160 parameters,
folded export equality and representative exact greedy outputs. Then submit
full4096/2000 origin Math reproduction and128x2048 AR/native/mapper evaluation.
A gate is not a full result; its output directory must stay separate.

### Task3: Transfer SpecForge adapter
Create transfer_model.py, transfer_data.py, train_transfer.py and tests.
Import the untouched vendor objectives.configure and pinned OnlineDFlashModel.
Replace only family-dependent dimensions, masks, checkpoint loading and F0
with explicit configuration. Preserve PEFT initialization, shard32 layout,
microbatch/RNG/DDP scheduling and2000-step horizon. Repack verified existing
dense caches with memory mapping; do not regenerate coherent target tokens
or recapture full features unnecessarily. Save a machine-readable delta
from the unmodified source. Test chunked versus unchunked AUF gradients,
invalid gaps, first failure, frozen tensors, checkpoint resume and export.

### Task4: Normal and ZIP comparators
Create train_normal.py using the existing RelaySpec normalization/loss.
Audit normalization epsilons and the exact historical parameterization;
test training-to-runtime equality, including input normalization which cannot
be absorbed into a constant dense weight. Reuse the existing ZIP trainer
and eligible checkpoints, preserving its two terms and position weighting.
Use the same4096 trajectories and write explicit data/compute/cost ledgers.
Do not substitute direct-fusion CE/AUF for the normal feature-loss baseline.

### Task5: Primary transfer jobs and runtime gate
Create matrix.json, run_transfer.sbatch, benchmark_transfer.py and collect.py.
First complete Q8 three-way comparison; overlap Q14/L3 work on remaining
capacity. Inference must verify deployed fusion, embedding/head and fused
context-KV weights. Repeat128x2048 measurements and pair output IDs before
reporting ratios. Add the single matched handoff-CE Q8 control.

### Task6: Cross-family and workload confirmation
Implement Task4's vocabulary gate independently while supported fits run.
Advance X8 only from verified bridge semantics. Evaluate the four frozen
development workloads; freeze choices, then run untouched confirmation.
Broader data/compute/size sweeps retain their original objective but are
reprioritized behind the exact handoff primary comparison. Final two
Transformers replications remain last, using Q8 and L3 if X8 is unsupported.

### Task7: Paper and reproduction package
Generate a separate handoff evidence table and claim ledger; retain the
previous AUF draft as a historical recipe investigation. Explain baseline
differences and initialization cost, include negative findings, and use
fresh measured controls. The supplied expected_results.json is a reference,
not our measured result. Full final manuscript/figures follow completed main
experiments; no speedup or cross-family support is anticipated as established.

## 6. GPU allocation and speed strategy

Maximum4 GPUs total. Keep paired2-GPU DDP fitting (exact handoff batch) and
up to2 independent one-GPU evaluation/capture workers. If two DDP fits run,
no additional GPU evaluator runs. Dependency chains enforce the cap.
All21 earlier pending arrays were held, not deleted, to prevent obsolete
recipe sweeps occupying the budget. The3 active older jobs finish normally.
Reuse frozen assets, shard32 caches, memory-mapped reads, bulk generation,
four-sequence capture, length bins, final fixed checkpoints, merged inference
weights and already-validated ZIP fits. Keep numeric precision and reduction
unchanged; no optimization may silently change the recipe.

## 7. Timing estimate and completion gates

These are planning ranges to replace after the first measured transfer gate:
- Preparation/fidelity gate: roughly30–90minutes if model downloads work;
  full origin labels must be regenerated because the author's caches are
  inaccessible. The handoff's3.83-minute Math fit excludes this work.
- Transfer fitting: initially budget10–30minutes per2-GPU fit, measured and
  revised after Q8. It is not the earlier three-epoch/four-anchor trainer.
- A128x2048 sequential pass at100–200TPS has a262144-token maximum,
  hence22–44minutes of generation at the cap; natural EOS usually lowers it.
  Include engine setup/warmup separately. Existing128 Numina passes contain
  about86k–123k tokens, giving roughly7–21minutes at those rates.
- First supported Q8 three-way table: approximately2–4hours after successful
  setup with cache reuse. Q8/Q14/L3 repeated main tables: approximately6–12
  hours. Four-workload breadth/confirmation and cross-family engineering take
  longer and are not honestly a few-minute job. No guaranteed cross-family ETA.

Completion requires: faithful golden checks, matched normal/ZIP/handoff
principal results, all128 outputs verified at2048 caps, repeated timing,
cost accounting, supported size/family coverage, a documented cross-family
outcome, required breadth/confirmation/profiling/backend checks and a rebuilt
audited manuscript. Pending jobs or passing unit tests do not satisfy these.

## Added architecture comparison: fusion residual versus five maps

User requested both parameterizations. At the same ZIP epoch-3 initialization,
compare handoff AUF with (a) frozen transfer fusion F0 plus rank-56 BA and
(b) the five independently trainable, bias-free W_i followed by frozen native
fusion F. Preserve the frozen native normalization in both cases. Handoff's
original parameterization includes F0; it is not BA alone.

| Transfer | Target feature width | F0+BA parameters | Five-map parameters |
|---|---:|---:|---:|
| Qwen4 drafter to Qwen8 | 20,480 | 1,290,240 | 52,428,800 |
| Qwen4 drafter to Qwen14 | 25,600 | 1,576,960 | 65,536,000 |
| Llama8 drafter to Llama3 | 15,360 | 1,089,536 | 62,914,560 |

Both use 4,096 records, 2,000 optimizer updates, effective batch eight,
eight anchors per record, unchanged AUF chunk/reduction/RNG/schedule, and
fixed final checkpoints. Initialization is checked against the same ZIP
export and its cost is reported separately. Trainable matrices are FP32;
forwards are BF16. Five-map training performs the original two-stage
projection; deployment folds it to one matrix. BF16 folding discrepancies
must be measured, not asserted to be bitwise absent. This is a capacity
comparison, not parameter-count matching.

Main decoding remains 128 requests, maximum 2,048 output tokens with natural
EOS, identical prompts and runtime, full token/finish agreement with AR,
and three timing repetitions. Report TPS, speedup over AR/ZIP/normal
RelaySpec when matched controls are complete, accepted progress, training
GPU hours and peak memory. Both variants deploy the same dense fusion shape.
Normal RelaySpec remains a distinct required control, not a synonym for ZIP.

First bounded Qwen8 gates: jobs 31551 (fusion_r56), 31552 (five_maps).
Each performs two optimizer updates and verifies frozen weights and export.
They wait for cache pack31528 and the older one-GPU evaluation31346, then
run sequentially beside the two-GPU origin reproduction31525; max four GPUs.
Full architecture runs follow successful numerical and decoding gates.


### Execution update

The initial pending gates31551/31552 were replaced before execution by
31553/31554 to add vLLM token/finish-exact deployment checks. Full fit jobs
31556–31561 now cover both architectures for Q8,Q14,Llama. Evaluation jobs
31562–31573 and CPU aggregation jobs are recorded in
`experiments/handoff_transfer/reports/{evaluation,collection}-jobs.json`.
Training is a serial two-GPU lane; evaluation is two serial one-GPU lanes.
The latter wait for the older runs to release their GPUs. Q8 uses the exact
original development manifest; its SHA256 matches the node06 manifest:
`c540fc20817109c468763ba8bc70d48472f5869c814585a977c7a7337fce4cc9`.
A missing node07 manifest path was linked to that existing immutable file.
The original handoff fit completed in231.266 seconds; long evaluation is
still running. This is evidence for fit efficiency, not transfer speedup.
