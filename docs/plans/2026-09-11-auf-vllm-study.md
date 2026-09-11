# AUF RelaySpec with vLLM: Research and Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evaluate whether accept-until-fail token supervision improves transferable speculative drafters, reproduce the important RelaySpec transfer and scaling studies with controlled baselines, and write a separate evidence-backed manuscript.

**Architecture:** Retain the ZIP's frozen target/drafter and exportable linear conditioning interface. Add a differentiable blockwise training path with the exact AUF mask, using vLLM for target rollouts, target-feature extraction, and primary inference. Separate a controlled loss replacement on the original mapper from the supplied rank-56 fusion-LoRA recipe, because they change different parameters.

**Tech Stack:** Pinned vLLM and DFlash sources, PyTorch autograd, BF16 matrix operations with FP32 loss reductions, safetensors, immutable JSONL manifests, Slurm on Turing, at most four GPUs, generated LaTeX tables/figures.

---

## 1. Status, scope, and inspected evidence

This is a plan, not a report of completed AUF experiments. No AUF training or evaluation jobs have been submitted. The user requested this plan before implementation.

- New worktree: `/home/aryamavmurthy/work/RelaySpec-AUF`.
- New branch: `research/auf-vllm-20260911`, starting at `f6b7d4bb` from the current paper worktree.
- Original current manuscript remains in `/home/aryamavmurthy/work/RelaySpec-scaling/paper/iclr2027/`.
- Separate new study: `experiments/auf_vllm/`; separate new manuscript: `paper/auf_iclr/`.
- Inspected ZIP: `/home/aryamavmurthy/work/RelaySpec/transfer.zip`, especially `README.md`, `IMPLEMENTATION.md`, and `src/train.py`.
- Inspected current paper source and generated family/rollout tables, plus the archived 128-question Llama and cross-family configurations.
- Turing SSH is reachable. The most recent queue check during planning showed no jobs for `aryama.murthy`. This is not a GPU reservation. Recheck before submission.
- Existing project Slurm scripts specify account `priyesh.shukla`, partition `u22`. Verify active association and resources before using them; prefer four matching L40S GPUs for continuity.

The ZIP's Qwen transfer used Qwen3-4B's drafter with Qwen3-8B, five linear maps, feature loss, 16,384 generated training records, three epochs, and vLLM 0.28.0+cu129. Its recorded fitting time was 887.55 seconds on one L40S. That is historical feature-only fitting time, not an AUF runtime estimate.

The current paper also contains Qwen3-14B, Llama-3.1-8B drafter to Llama-3.2-3B, Qwen3-4B drafter to Llama-3.1-8B, and EAGLE-3 studies. The archived full Llama/cross-family evaluations used a different runtime and FP32 configuration. Their TPS must not be used as the new vLLM loss baseline without rerunning.

User-specified wording for the experimental scope and new manuscript:

> Our reported Qwen3 experiments use a shared tokenizer and token-ID vocabulary; the mapper adapts hidden representations or the fusion interface, not vocabulary IDs. Consequently, these results do not demonstrate heterogeneous-vocabulary DFlash support.

This statement describes the reported Qwen3-to-Qwen3 experiments. The planned Qwen-to-Llama experiments are a separate, unvalidated extension of the new vLLM/AUF study; their vocabulary compatibility cannot be inferred from the Qwen3 results.

## 2. Questions the experiments must answer

1. Does supervising the correct prefix and first failure improve actual accepted progress and throughput relative to ordinary token CE?
2. Does token supervision improve on the ZIP's layer-plus-context feature loss at comparable data and training cost?
3. Can AUF learn a useful interface from scratch, or is feature-based initialization important?
4. Is the supplied rank-56 fusion update sufficient, and how does the answer change with target size and model family?
5. How do distinct records, optimization work, and mapper capacity interact?
6. Do improvements survive unseen workloads, new training seeds, different request batch sizes, and the final Transformers replication?
7. Does any extra fitting cost amortize in a realistic number of generated tokens?

The target is a resolved useful gain; a 10% uplift is an aspiration, not an assumed result or a reason to exclude unfavorable cells. Report wins, ties, regressions, and unsupported combinations.

## 3. Define the methods before running them

### 3.1 Exact AUF computation

For logits `[blocks, positions, vocabulary]`, labels and validity mask `[blocks, positions]`:

```python
correct = logits.detach().argmax(-1).eq(labels)
support = (~valid) | correct
inclusive = support.to(torch.int64).cumprod(-1)
exclusive = torch.cat([torch.ones_like(inclusive[:, :1]), inclusive[:, :-1]], -1)
active = valid & exclusive.bool()
# CE is computed only at active indices; invalid padding labels never enter CE.
loss = cross_entropy(logits[active].float(), labels[active], reduction="sum") / active.sum()
```

This is the semantic reference; the optimized implementation may chunk the vocabulary/positions only after value and gradient equivalence tests.

- Recompute the mask from the CURRENT drafter at every forward pass; do not cache correctness from an older/native drafter.
- Include the first incorrect proposal. Exclude all positions after it.
- Exclude clean anchor, padding, and positions beyond the recorded response/EOS.
- Preserve the user's per-microbatch normalization. With gradient accumulation over G valid microbatches, backpropagate each normalized loss divided by G. Do not silently replace this with one global active-token denominator.
- Reject malformed all-invalid microbatches before an optimizer update; account for any dropped incomplete batch explicitly.
- No MSE, KL, positional decay, or added loss coefficient in the AUF phase.
- Only intended interface parameters update. Frozen drafter weights still participate in autograd to propagate gradients to the interface; wrapping the drafter forward in `no_grad()` would break learning.
- AUF is a surrogate for greedy accepted progress, not a proof of higher speed or an exact gradient of end-to-end latency.

### 3.2 Track A: controlled replacement of the ZIP loss

Retain the same five trainable bias-free layer maps `W_i`, frozen fusion `F`, frozen RMSNorm, frozen drafter, and fixed embeddings/head. Train three arms from identical initial tensors:

| Arm | Trainable weights | Training objective | What the comparison establishes |
|---|---|---|---|
| ZIP-FEATURE | Five `W_i` | Original equal-weight layer + context relative MSE | Original ZIP recipe control |
| TOKEN-CE | Same five `W_i` | Uniform valid-position token CE | Value of token supervision |
| TOKEN-AUF | Same five `W_i` | Exact AUF | Incremental value of the prefix mask |

AUF versus CE is the cleanest loss comparison: same rollouts, block inputs, initialization, batches, updates, head, and optimizer search budget. ZIP versus token loss uses the same records, but the objectives consume different supervision and incur different work. Report that distinction and include both matched-exposure and matched-GPU-time comparisons.

The original paper's normalized direct-context loss is an additional historical control for the two primary Qwen settings. Do not call it identical to the ZIP's two-term loss.

### 3.3 Track B: reproduce the supplied fusion-LoRA recipe

After obtaining a dimensionally compatible frozen interface `F0`:

`F_new = F0 + (alpha/r) B A`, with `r = alpha = 56`.

- Initialize A using a recorded seed and B to zero, so the initial function is exactly F0.
- Train only A and B; freeze F0 and all other weights.
- Compare frozen F0, F0+CE-LoRA, and F0+AUF-LoRA using identical starts and budgets.
- For transfer, propose using the ZIP-exported folded interface as F0. This is a two-stage recipe: **ZIP initialization followed by pure AUF refinement**. Charge the ZIP initialization to total cost and to total distinct-data exposure.
- This is not the same experiment as replacing the ZIP loss on five randomly initialized `W_i`.
- The source-native fusion matrix generally has the wrong input width after transfer; adding a same-shape LoRA update cannot fix a width mismatch. Never silently slice or pad it.
- Example Qwen4→Qwen8: folded F0 is `[2560, 20480]`, so rank-56 A/B have 1,290,240 trainable parameters. The ZIP's five maps have 52,428,800. Report both trainable parameters and deployed matrix size.
- Merge A/B into F0 for inference and verify folded/unfolded outputs. Trainable rank changes adaptation cost; a merged residual does not by itself shrink the deployed dense matrix or guarantee lower inference cost.

### 3.4 Two inputs requiring resolution at the implementation gate

1. The supplied AUF definition specifies **LoRA-adapted target rollouts**. The inspected ZIP instead uses an unadapted frozen target. Locate and pin the intended target adapter/checkpoint and its rollout provenance before claiming exact reproduction. The teacher generating labels, the model supplying target features, and the verifier/AR baseline must be the same adapted target. If no such adapter exists, report this dependency explicitly; a base-target AUF extension must be labeled separately, not passed off as that recipe.
2. Interpret “along with ZIP” as using its transfer architecture/data/export path, with pure AUF during token refinement. Do not invent `AUF + lambda*MSE`. Any simultaneous combined objective would be a separately named later ablation requiring an explicit protocol revision.

Neither dependency prevents writing the plan, auditing data, or implementing and testing the generic AUF function.

## 4. Model and runtime matrix

An arrow always means **drafter's original source target → new verification target**.

| ID | Source of released drafter | New target | Role |
|---|---|---|---|
| Q8 | Qwen3-4B DFlash | Qwen3-8B | First pilot; complete scaling reference |
| Q14 | Qwen3-4B DFlash | Qwen3-14B | Larger target replication |
| L3 | Llama-3.1-8B DFlash | Llama-3.2-3B | Llama same-family, smaller target |
| X8 | Qwen3-4B DFlash | Llama-3.1-8B | Attempt cross-family feasibility first; full study conditional on validation |
| E8/E14 | Existing Qwen-source EAGLE-3 | Qwen3-8B / Qwen3-14B | Original paper's second drafter architecture |

These cover the original paper's important target sizes. Add Qwen4→Llama3 as a cross-family target-size extension after X8 passes. Do not create a full Cartesian product of every available model size. New source drafters or 32B/70B targets are outside this first replication matrix.

For every row pin model/tokenizer IDs and revisions, selected layers, vocabulary contract, embedding/head provenance, target adapter if applicable, context normalization, and draft length. For Qwen3-to-Qwen3, record shared tokenizer/token-ID compatibility and no vocabulary remapping; record an explicit bridge only for a separately validated heterogeneous-vocabulary configuration. The Llama models currently archived are the specific `unsloth/Llama-3.1-8B-Instruct` and `unsloth/Llama-3.2-3B-Instruct` revisions; do not substitute another publisher's weights under the same short label.

DFlash AUF is the initial implementation. EAGLE's autoregressive proposal path must use its own correct runtime-conditioned prefixes; applying a parallel DFlash mask to teacher-forced EAGLE logits is not an equivalent experiment. Develop and validate that extension after the DFlash core, then repeat the Qwen8/14 transfer comparison.

## 5. vLLM integration gates

Start from the ZIP-pinned vLLM 0.28.0+cu129 environment, verify the actual installed source and wheel hashes inside an allocation, and freeze the working stack. If a newer revision is necessary, test it in a separate environment and rerun every matched arm on that revision. Never compare an AUF result from a new engine with historical baseline TPS from another engine.

vLLM handles batched rollout generation, frozen target-feature capture, and primary decoding. The AUF optimizer uses a differentiable PyTorch DFlash training module with equivalent block semantics and weight export; do not describe ordinary `LLM.generate` as providing backpropagation. This training module is distinct from the standalone Transformers inference replication deferred until the end. vLLM's Speculators training integration is a candidate to inspect and reuse, not an assumed drop-in AUF trainer.

Primary-source integration references checked during planning:

- [vLLM speculative decoding](https://docs.vllm.ai/en/v0.28.0/features/speculative_decoding/): native/custom proposer interfaces and feature restrictions. Its documented heterogeneous-vocabulary switch is restricted to `method=draft_model`; it is not a DFlash cross-family switch.
- [vLLM batch invariance](https://docs.vllm.ai/en/v0.28.0/features/batch_invariance/): reference for numerical configuration; verify compatibility with the pinned attention backend and graphs.
- [Speculators DFlash training integration](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md): candidate training/export integration, to be pinned and tested.

### Same-family gate

For Qwen3-to-Qwen3, verify the shared tokenizer and token-ID vocabulary, including special tokens. Keep token IDs unchanged: the learned map changes hidden representations or the fusion interface only. This gate is not a test of heterogeneous-vocabulary support.

Verify hidden tap indices, positional offsets, block-attention semantics, frozen source embeddings/head, mask token, target logits, KV truncation, EOS, and folded checkpoint export. A drafter input contains the true anchor and masked future positions, never the future gold tokens being predicted. Conditioning features may contain only the context available at that draft step.

### Cross-family gate

The existing Qwen→Llama code has explicit tokenizer bridging. Port and audit that behavior in a registered DFlash/custom proposer rather than silently swapping in an independent draft transformer.

- Define source/target token IDs and byte spans explicitly. Equal integer IDs across vocabularies do not establish equivalent tokens.
- AUF correctness and CE must refer to the actual proposal distribution in the space being verified.
- Test representable labels, multi-token boundary changes, many-to-one mapping, unmapped tokens, Unicode, whitespace, code indentation, and EOS.
- A frozen source head cannot assign positive probability to every target token merely by renaming token IDs. If the bridge lacks support for a target label, do not clamp its infinite CE, map it to an arbitrary token, or mask it away as padding.
- First establish a supported, normalized probability mapping and its relationship to runtime proposal decisions. If that cannot implement the supplied loss exactly, document a vocabulary-aware extension and its coverage as a separate method. Exact cross-family AUF remains gated; do not claim it has been reproduced.
- Any new vocabulary/head adapter must be identical across ZIP, CE, and AUF controls and disclosed as an architectural change. There must still be no live source transformer in the relay inference path.

Cross-family integration can proceed alongside same-family experiments within the four-GPU cap. It cannot be replaced by an early Transformers benchmark just to produce a number.

### User-authorized cross-family decision rule

Attempt the required cross-family engineering and validation. If it is not feasible within this method, use the exact shared-vocabulary wording in Section 1 and scope the manuscript's new evidence to the validated configurations. No further approval is needed to take that fallback.

1. Audit the existing tokenizer bridge and its relationship to per-position AUF labels. Static inspection found a source-token decode/target-token re-encode path, with optional token-ID lookup shortcuts. This produces proposals for greedy verification; it does not by itself define a differentiable target-vocabulary distribution for AUF. See `experiments/auf_vllm/reports/cross_family_static_audit.json`.
2. Prototype the smallest compatible vLLM proposer/training contract. Check normalization/support, label positions, boundary changes, anchors, EOS, and source-free deployment before expensive capture or fitting.
3. Run a bounded diagnostic pilot, then a 128-request development evaluation with a 2,048-token cap if the implementation passes. Apply the same vocabulary mechanism to all loss controls. Do not start the large cross-family sweep before that gate.
4. If exact AUF requires a materially different learned vocabulary/head architecture, or training/inference alignment cannot be validated, record the concrete limitation and defer that separate extension. Do not invent a surrogate and label it the supplied AUF.
5. On this fallback, omit X8 and its dependent scaling/serving rows from the new manuscript's empirical claims, retain the attempt and reason in the experiment ledger, use L3 for the planned second Transformers replication, and continue Q8/Q14/L3 studies as validated. The Qwen3 results retain precisely the shared-vocabulary scope supplied by the user.

This fallback is for compatibility or correctness limitations, not disappointing performance. A valid cross-family experiment that yields no speedup must still be reported as a result. Historical cross-family results retain their original runtime and scope and are not rewritten as new AUF evidence.

## 6. Data protocol and cache design

### Splits

- Build immutable train, validation, development, and confirmation manifests before sweeping.
- Main calibration source: pinned NuminaMath-CoT selection, using new-target rollouts, consistent chat template, and `enable_thinking=False` for applicable Qwen targets.
- Preserve the old 128-question manifests as explicitly exposed legacy replication sets. They were already used repeatedly for research; they are not fresh tests.
- Reserve 1,024 records for offline feature/token validation and 128 distinct requests for development decoding/tuning.
- Create a new untouched 128-request confirmation manifest per principal workload where sufficient independent records exist; never select checkpoints, block size, rank, or learning rate on these requests.
- Audit normalized exact and template/group overlap among calibration, development, confirmation, and public benchmarks. Record source revisions, group IDs, counts removed, and remaining near-overlap limitations.
- If a public workload has already been fully exposed, disclose that and add a fresh compatible source for confirmation rather than claiming unseen data.

### Training records, lengths, and epochs

- A record means one distinct original problem/conversation, not a token, sampled position, anchor, chunk, or repetition.
- Data counts: **16, 32, 64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384, 32,768**.
- Create nested, source-stratified subsets from a single ordered 32,768-record manifest. Extend the ZIP's 16,384 selection without consuming its held-out groups. The added selection is a new dataset realization, documented as such.
- Generate continuations once per distinct target checkpoint; reuse them across losses, capacities, and epochs when their hashes match.
- Main rollout cap: 4,096 generated training tokens, with prompt cap 1,024, matching the richer ZIP recipe. Evaluation cap remains 2,048.
- Treat a shorter 512-token training recipe only as a separately named length/cost ablation; never substitute it silently to make AUF faster.
- Proposed token-training epoch: visit every selected record once and sample four valid response anchors per record, deterministically with an epoch seed. Sample without replacement when possible; for very short records log actual block counts rather than counting padding as data.
- Log selected records, actually visited distinct records, full/padded tokens, supervised block opportunities, active AUF tokens, optimizer steps, and elapsed GPU time. An epoch of token blocks is not numerically equivalent to an epoch over the ZIP's sampled features.

### Batched preparation

- Start vLLM generation at 64 active sequences/GPU, 8,192 batched tokens, and 512 queued prompts/call; benchmark 32/64/128 active sequences during a bounded pilot.
- Freeze the chosen generation scheduling configuration before creating the canonical rollouts. Scheduling is part of data provenance.
- Use length buckets and atomic 128-record output shards, with deterministic record-to-worker assignments and resume checks.
- Frozen teacher-forced capture starts at 8 sequences/call, testing 4/8/16 subject to memory and padding. Persist BF16 raw features with token positions, sequence IDs, label IDs, source hashes, and completion markers.
- Stage shards on node-local scratch with bounded prefetch and pinned host buffers. Reuse read-only caches across jobs; do not copy hundreds of GB per fit.

### Critical difference from the ZIP cache

The ZIP saved approximately one quarter of token positions, enough for independent feature regression. DFlash block training may need contiguous conditioning-prefix features. Random quarter-density tensors cannot be treated as full-context DFlash input.

Capture the context needed by the actual block input. Cache frozen raw target features, not mapper outputs or draft K/V that change when the mapper updates. AUF does not require the original source transformer for label generation or token training; the ZIP feature baseline still requires its aligned feature targets.

Storage gate: use the observed full token count to calculate `2 * tokens * sum(tap_widths)` bytes. For the ZIP's approximately 21.6M tokens, five dense 8B target taps alone are approximately 0.89 TB before metadata. This is substantially larger than a sampled target cache. Check scratch capacity and throughput before dense capture; use bounded shard staging/recomputation when needed and report the recomputation cost. Do not change context length or sampling semantics silently to fit disk.

## 7. Training batching and optimizations

Proposed AUF/CE starting batch: four records per microbatch, four anchors per record (up to 16 draft blocks), eight accumulation steps, giving 32 record-visits and up to 128 blocks per optimizer update. This is a pilot starting point, not an already measured optimum.

1. Sweep microbatch records 1/2/4/8 on Q8 with fixed valid logical batches; measure peak memory, blocks/s, and gradients. Freeze the selected microbatch/accumulation contract per matched pair before fitting.
2. Bucket by prefix length. Share context within a record where the implementation supports it without future leakage; use block-diagonal/segmented attention so anchors cannot see another block's hidden future context.
3. BF16 frozen weights and matrix operations, FP32 trainable master weights, CE reductions, and optimizer state where appropriate. Verify finite values and clip global trainable gradient norm at 1.
4. Fused AdamW, weight decay zero for the main study. Equal small learning-rate search for AUF and CE: `1e-4, 3e-4, 1e-3` on development only; 5% warmup and a declared schedule. Preserve ZIP's `1e-3` reference and give it a matched tuning allowance in the tuned comparison.
5. Chunk exact vocabulary projection/CE to bound memory. Avoid full probability tensors if exact logsumexp and greedy argmax can be obtained equivalently. Do not use sampled softmax or top-k approximate CE in the primary result.
6. Use activation checkpointing only when it improves feasible throughput; compare with the uncheckpointed gradient reference and log its compute cost.
7. Test `torch.compile` after the eager trainer is correct. Cache stable shapes; measure compilation separately; count cold-start cost in total preparation cost. Do not enable graph/compile changes in one comparison arm only.
8. Save one continuation trajectory at declared update/epoch checkpoints and evaluate those; do not repeatedly restart equivalent fits. Some fresh endpoint fits are still needed if cosine schedules differ with planned total steps.
9. Use four independent single-GPU workers for fits/seeds when models fit. Use TP2 only if necessary for a target and apply it equally to its controls; two TP2 engines exhaust the four-GPU cap.

Frozen weights reduce optimizer cost but AUF still backpropagates through the drafter and vocabulary projection. Establish an AUF-specific time estimate; the ZIP mapper-only timing is insufficient.

## 8. Experiment matrix and priorities

Do not take a Cartesian product of every axis. The rows below define separate controlled studies, with one shared reference configuration per pair.

All X8-dependent rows are conditional on the cross-family decision rule above. Their presence in this matrix is an intended experiment, not a claim of supported heterogeneous-vocabulary DFlash.

| Study | Planned settings | Controls and evaluation | Purpose |
|---|---|---|---|
| End-to-end correctness | Q8, then Q14/L3/X8; 16–64 training records | Small diagnostic prompts, then 128 development requests × 2,048 cap | Prove the actual path works before scale |
| Main transfer | Q8, Q14, L3, X8; N=4,096; 3 epochs | Track A's three losses; Track B's frozen/CE/AUF starts; AR and native where available | Principal table |
| Main seed replication | Seeds 42, 43, 44 at frozen N=4,096 configuration | Same initialization seed paired across AUF/CE/ZIP where architectures match | Fit variability without cherry-picking |
| Data scaling, fixed updates | Q8: all 12 N values; X8: 512/2,048/4,096/8,192/16,384/32,768 | Initially AUF/CE at 1,024 updates; ZIP matched record-exposure series separately | Distinct data at declared work |
| Data scaling, fixed epochs | Same Q8 grid; X8 six points; Q14/L3 at 512/4,096/16,384 | 3 epochs, all Track A losses; same selected records | Benefit when larger datasets also get more exposure |
| Epoch scaling | Q8 and X8, N=512/4,096/16,384 | Epochs 1/3/6/12 for AUF/CE/ZIP; Q14/L3 endpoints 3/6 | Underfitting, saturation, overfitting |
| Update trajectories | Q8 at fixed N=4,096 | 128/256/512/1,024/2,048/4,096 updates, continuous schedule | Compute curves without extra fitting |
| Rank and parameterization | Residual ranks 8/16/32/56/112/224, alpha/r=1; dense reference | Q8 at N=512 and 4,096; AUF/CE; confirm rank32/56/112 on X8/L3 | Adaptation capacity versus cost |
| Architecture control | ZIP five maps versus directly trained folded fusion; factored linear bottlenecks128/512/1,024/2,048; MLP widths256/1,024/4,096 | Q8 at N=512 and4,096; frozen data and compute; AUF/CE; actual parameter counts | Whether linearity remains sufficient under token loss |
| Initialization | Random maps versus matched-N ZIP initialization; native columns only when dimensionally valid | Q8 and one non-Qwen target; matched CE/AUF continuations | Whether AUF's benefit depends on warm start |
| Training-composition generalization | Math-only versus fixed-count mixed math/code/dialogue | N=4,096; Q8 and X8; AUF/CE/ZIP | Transfer outside calibration domain |
| Regularization check | Weight decay0/0.01/0.1 on Q8 at N=512; rank56 and dense | AUF/CE, equal budgets; held-out validation and128-request development endpoints | Whether small-data overfitting changes with the loss |
| Draft length | Block sizes 4/8/10/16/24/32, restricted to supported positional ranges | Tune all arms on development; main fixed-block and separately tuned results | Acceptance/verification cost tradeoff |
| Serving concurrency | 1/4/8/16/32, max supported without OOM | Frozen selected Q8 and X8 variants plus same-runtime controls | Does speed survive batching? |
| Secondary drafter | EAGLE-3 Q8/Q14 after its AUF semantics pass | Uniform CE, original interface loss, AUF, native and AR | Mechanism generality |
| Final Transformers replication | Q8 and X8; if X8 remains incompatible, predeclared L3 alternative | Same frozen checkpoints, 128 requests/workload, 2,048 cap | Backend dependence, performed last |

For fixed-update AUF/CE, 1,024 × 32 record-visits allows every record in the 32,768 pool to be visited once. The sampler must enforce coverage; report actual counts. Smaller pools necessarily repeat more. Keep the logical record batch fixed within this sweep. If memory changes microbatch size, preserve and disclose the per-microbatch loss contract; do not describe changed aggregation as exactly identical optimization.

ZIP's historical optimizer batch is 2,048 sampled positions, with equal-example weighting. Its original 1,024-update sweep used four examples/update. Preserve that historical curve as a separately labeled reproduction; add a new record-exposure-matched ZIP protocol rather than pretending those different update definitions are equal compute. Compare all losses at measured GPU-time budgets as well.

Initial sweeps use seed42 and are exploratory. The principal transfer configuration and selected low/high-data confirmations use three fit seeds. Training repetitions and runtime repetitions are distinct. Every plotted decoding endpoint uses 128 development requests and the 2,048 cap; tiny diagnostics never substitute for a plotted endpoint. Use unopened confirmation data only after the selection rules and configuration are frozen.

Keep epoch12 even if it regresses; numerical failure stops a cell with an explicit record. Extra epochs beyond12 need a new justified protocol revision, not an indefinite search for a win.

For capacity controls, the input/output dimensions are fixed by the two models. Change rank/bottleneck width or parameterization rather than changing hidden dimensions arbitrarily. A product of linear matrices can be folded into a dense deployment matrix; measure both the folded and factorized implementation on development before freezing the export choice. An MLP with an activation cannot be algebraically folded into that matrix, so include its actual inference overhead and parameter count. Residual rank, zero-base low-rank mapping, and nonlinear MLP capacity are different experiments.

For the mixed-data cell, use2,048 math +1,024 code +1,024 dialogue training records, generated by the same new target, against4,096 math records. Equalize record visits and report token-length distributions; do not use evaluation tasks as training content. Weight decay is an optimizer ablation and does not add an MSE/KL term to AUF; any explicit extra loss is outside the main objective.

## 9. Evaluation contract

### Request sets

- Every principal per-workload AUF comparison: **128 distinct requests per method/checkpoint**, **maximum 2,048 generated tokens**, identical prompts and termination policy across arms.
- Workloads: MATH, GSM8K, code, and dialogue. Main transfer breadth uses 128 each where a valid set exists, not 32 of each disguised as a 128-per-workload evaluation.
- Retain full original-paper benchmark coverage for frozen representative Q8/Q14 final variants: MATH-500 (500), GSM8K (128), HumanEval (164), MBPP (378), and MT-Bench (80 two-turn conversations). Deduplicate requests already evaluated under identical conditions. MT-Bench remains 80 conversations; do not fabricate 128 unique items.
- The new 128-per-workload confirmation and full historical/public suites serve different evidence roles. Keep their exposure history visible.
- Cap means natural EOS can finish earlier. Record actual output tokens, finish reason, and fraction capped. Never force 2,048 tokens by disabling EOS for the main results.
- Ensure model length accommodates the actual prompt plus 2,048 output and proposal lookahead. Account for longer second-turn dialogue context; no silent truncation.

### Timing and batching

The main paper measures one sequential request per GPU worker. Preserve that primary metric and parallelize independent request shards across up to four GPUs. With four workers, each covers 32 of a 128-request set.

Separately benchmark serving at concurrency 4/8/16/32. Use a fixed prompt order/arrival protocol for all methods, record aggregate tokens/wall-second, request latency, TTFT, and inter-token latency. Never divide a synchronous batch's wall time by the number of requests and call the result single-request latency.

- Batch-one primary runtime: same target, precision, tokenizer, graph mode, attention backend, KV policy, and sampling settings for all methods.
- Greedy temperature0; no top-p/top-k truncation or repetition penalties; model-appropriate stop IDs; thinking disabled where applicable.
- Prefix caching off in the primary comparison to avoid repeated-prompt cache advantages. A cached-serving configuration can be a separate systems study.
- Use the ZIP's batch-invariant, no-compile reference settings with supported CUDA decode graphs as the starting correctness configuration. Benchmark further optimizations equally for every method after validation.
- At least four excluded representative warmup prompts and coverage of timing shapes. Save cold setup/compile times; follow one declared retry policy for compilation-affected requests.
- Three rotated hardware/order repetitions for principal claims. Fixed checkpoints, request set, and controls; avoid colocated training on any timed GPU.
- Run methods in isolation for memory accounting. GPU rotation balances device effects; record power caps, clocks, temperature, and concurrent system activity.

### Controls

1. Same-target AR in the new runtime.
2. Released target-native DFlash/EAGLE where a compatible checkpoint exists. Mark absent controls N/A.
3. Original ZIP feature mapper rerun on the same engine/hardware.
4. Uniform token CE with matched architecture and budget.
5. AUF.
6. Frozen initialization for refinement studies.
7. Source-assisted control for representative source-removal/memory and break-even measurements.
8. An appropriate independent draft-model baseline in vLLM for the main target, with equal tuning budget, where available.

Retain PARD, SD², and other previously studied methods in a positioning table with their actual adaptation/runtime assumptions. Only make direct speed comparisons when target, prompts, output cap, precision, runtime, batch regime, and hardware are matched. Otherwise present separately labeled reference results. Literature verification must establish which methods really address this transfer setting.

### Metrics and statistics

Record raw per-request prompt IDs, generated token IDs, elapsed time, actual tokens, stop reason, verification steps, proposed/accepted tokens, accepted-prefix histogram, and per-position acceptance.

- Batch-one TPS: `sum(output_tokens) / sum(request_seconds)`.
- Speedup over AR and over ZIP/CE: ratio of paired sums of request time. TPS and time ratios coincide only when output totals agree.
- Native-throughput retention when a native control exists.
- Latency median/p95, TTFT/ITL for serving, and cold versus warm setup cost.
- Exact complete-token-array agreement with AR, including length and stop; first divergence and prefix match distribution if disagreement occurs.
- MATH/GSM answer correctness, code execution tests/pass@1, and dialogue output/termination quality using a frozen declared scoring protocol. Do not present unjudged dialogue as measured accuracy.
- Peak process allocated/reserved GPU memory, device memory, training/inference energy where telemetry supports it, and total fitting GPU-hours.
- Paired 95% request bootstrap intervals, resampling requests with repeats kept together; equal-workload aggregate plus per-workload values. Fit seeds are an additional variation source, not additional independent requests.
- Predeclare AUF-versus-CE and AUF-versus-ZIP principal contrasts; treat large sweep endpoints as exploratory, report all cells, and use multiplicity-aware inference for families of formal claims.

No universal exact-match guarantee is assumed from BF16. A correctness failure blocks promotion to a lossless headline result. Diagnose identical-prefix target logits and cache states, fix the implementation or adopt a common validated numerical mode for all arms, then rerun. Do not quietly relax verification or score only agreeing requests. A FP32 diagnostic must not be mixed with BF16 baseline timings.

## 10. GPU profiling and total cost

Every job: Slurm job ID, GPU UUID/model, allocation, visible devices, source/config/checkpoint hashes, versions, start/end times, GPU utilization, memory, power, temperature, clocks, and errors.

- Sample low-overhead GPU telemetry approximately every second, with coverage/missingness checks.
- Profile representative AR, native, ZIP, and AUF on Q8; repeat the key conditioning/verification comparison on L3 or X8.
- Use Nsight Systems for CPU launches, transfers, target prefill/decode/verification, drafter, interface, and vocabulary bridge. Training trace separates data wait, forward, vocabulary projection, backward, optimizer, and checkpoint I/O.
- Use bounded Nsight Compute samples for representative mapper/draft/verification kernels: occupancy, compute utilization, supported DRAM/L2 traffic, and bandwidth. Unsupported metrics are N/A.
- Profiled timings stay separate from headline throughput. Recheck outputs and acceptance after instrumentation.
- Report rollout generation, target/source feature extraction, fitting, export, compilation, and inference separately. Cache reuse helps iteration but does not erase first-time deployment cost.
- Break-even generated tokens: total incremental adaptation seconds divided by positive per-token inference time saving; label hardware assumptions. If AUF is slower, report no finite break-even under that setting.

## 11. Four-GPU execution schedule

Use one campaign controller with an allocation ledger covering all this user's active study jobs. Never submit a second four-GPU allocation while another is running. Prefer a single four-GPU allocation with one isolated worker per GPU, or a globally throttled job array with equivalent accounting.

| Phase | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| Correctness | Q8 AUF end-to-end | Q8 reference/export equivalence | Llama/runtime microtests | Vocabulary bridge/mask microtests |
| Data production | Batched target rollouts | Batched target rollouts | Feature capture from completed shards | Feature capture/ZIP source targets |
| Main fitting | AUF fit | Matched CE fit | ZIP feature fit | Rank56 AUF/CE continuation or next seed |
| Scaling | AUF data/epoch cells | CE matched cells | ZIP/capacity cells | Non-Qwen confirmation cells |
| Primary evaluation | Request shard0 | Request shard1 | Request shard2 | Request shard3 |
| Serving/profile | Isolated serving/trace | Independent measured cell if no interference | Independent measured cell if no interference | Independent measured cell if no interference |
| Final backend check | Transformers Q8 | Transformers X8/L3 | Frozen vLLM paired reference | Verification/scoring as GPU need permits |

During early iteration keep two workers on bounded diagnostic tasks targeting under ten minutes of experiment compute after setup. They test correctness and engineering changes. Full 128×2,048 evaluations are separate jobs and are not promised to finish in ten minutes. Use watchdogs to stop hangs, not to discard slow valid requests.

Stage data ahead of the GPU consumer; do not leave four GPUs allocated for long CPU-only planning. Atomic completion manifests, resumable shards, strict nonzero exits, and checkpoint-compatible resume prevent silent partial success.

Rotate method/device assignments between primary evaluation repetitions. No training runs on timed GPUs; shared host-memory, CPU, power, and disk interference must also be excluded or the measurement rerun under the declared protocol.

## 12. Implementation tasks and verification gates

Paths below are new study files unless explicitly identified as a read-only reference. Commands describe the implementation stage; these files are not claimed to exist yet.

### Task 1: immutable protocol and source inventory

Files: `experiments/auf_vllm/protocol.yaml`, `models.json`, `inventory.py`, `tests/test_protocol.py`.

1. Write tests rejecting GPU caps above4, unknown loss IDs, missing revisions, conflicting train/eval IDs, and untracked changes to frozen manifests.
2. Run `python -m pytest experiments/auf_vllm/tests/test_protocol.py -q`; expect missing-module failure before implementation.
3. Implement strict configuration validation and source/ZIP checksum inventory; pin resolved target adapters and runtime.
4. Rerun tests; expect pass. Commit only these study files.

### Task 2: AUF and freeze correctness

Files: `losses.py`, `interfaces.py`, `tests/test_auf.py`, `tests/test_freezing.py` under `experiments/auf_vllm/`.

1. Write independent hand-calculated tests for `[correct, correct, wrong, correct, wrong] -> [1,1,1,0,0]`, first wrong, all correct, anchor/padding, EOS, zero-active handling, and independent blocks.
2. Check analytic logit gradients against masked CE; post-failure logits must receive zero direct CE gradient. Verify no gradient flows through mask construction.
3. Test that accumulated per-microbatch means differ from a global-token mean on an intentionally unequal-support example, and preserve the requested former behavior.
4. Implement AUF and rank56 fusion interface. Check frozen-weight hashes, nonzero intended gradients, zero-update initialization equivalence, and finite optimizer steps.
5. Run targeted tests, then commit.

### Task 3: block builder and shared data

Files: `prepare.py`, `capture.py`, `blocks.py`, `cache.py`, `tests/test_blocks.py`, `tests/test_cache.py`.

1. Test clean anchor/label shifts, no future conditioning, no cross-block/record attention leakage, padding positions, exact sampled-record accounting, shard resume, and corrupted-hash rejection.
2. Adapt inspected ZIP data paths; use vLLM for target rollouts and capture. Existing root `experiments/native_joint/core.py` is a block-layout reference, not an AUF implementation to adopt unchanged.
3. Compare batched block outputs/gradients with independent one-block references, including variable prefix lengths.
4. Run a bounded actual-GPU capture/train/export pilot; record performance and storage estimates. Commit after tests and artifact checks.

### Task 4: trainer and efficient export

Files: `train.py`, `export.py`, `tests/test_train_resume.py`, `tests/test_export.py`.

1. Test deterministic resume including optimizer/RNG/epoch/anchor stream, exact trainable-parameter sets, objective IDs, and data lineage.
2. Implement eager baseline with the declared microbatch normalization, then measured batching/chunked CE/compile optimizations.
3. Verify optimized loss/gradients against the eager reference within declared dtype tolerances.
4. Export frozen source embedding/head and folded interface; compare outputs to unmerged training model, hash all tensors, and verify the source trunk is absent at inference.
5. Commit the passing path before any broad sweep.

### Task 5: vLLM inference and cross-family support

Files: `vllm_runtime.py`, `vocab_adapter.py`, `tests/test_vocab_adapter.py`, `tests/test_runtime_contract.py`.

1. Build Q8 baseline/AUF load and generation tests on diagnostic prompts.
2. Implement and test the supported vocabulary mapping contract; explicitly fail unsupported cross-family loss semantics.
3. Validate full exported computation, cache cropping, EOS and greedy target decisions; run Q8 128-request development confirmation before launching scale.
4. Repeat compatibility gates for Q14/L3/X8. Record each gate independently and commit passing integration.

### Task 6: scheduler and experimental execution

Files: `matrix.py`, `campaign.py`, `run.sbatch`, `STATUS.json`, `tests/test_campaign.py`.

1. Write tests for resource ledger <=4 GPUs, dependency gates, complete128-row merges, no duplicate request IDs, and no test-set tuning input.
2. Generate explicit unique experiment IDs from family/loss/init/data/epochs/rank/seed/runtime; deduplicate shared endpoints and checkpoints.
3. Implement submission/monitoring/resume with failure manifests and immutable job configurations.
4. Run AUF correctness first, then required controls and transfer fits, then scaling and frozen confirmations in the priority order above.

### Task 7: evaluation, analysis, and profiling

Files: `benchmark.py`, `score.py`, `analyze.py`, `profile.py`, `tests/test_metrics.py`, `tests/test_evidence.py`.

1. Test ratio-of-sums, paired resampling, incomplete-run rejection, repeat mismatch, capped outputs, and missing telemetry.
2. Execute declared128-request evaluations and seed/repeat confirmations. Record all tested endpoints.
3. Complete main benchmark breadth, serving batch study, and representative GPU traces.
4. Produce machine-readable evidence tables and figures; no manual transcription of winning TPS.

### Task 8: final Transformers replication

Files: `transformers_reference.py`, `backend_comparison.json`.

1. After the vLLM campaign and configuration freeze, load exactly the selected exported checkpoints.
2. Run Q8 plus X8 (or the predeclared L3 compatibility alternative) on128 requests/workload with2048 output cap.
3. Use each backend's own matched AR/ZIP/CE/AUF controls; report numerical and performance differences. Do not retune on confirmation results.

### Task 9: separate manuscript and reproducibility package

Files: `paper/auf_iclr/main.tex`, `references.bib`, `generated/`, `figures/`, `experiments/auf_vllm/build_paper_assets.py`, `claim_evidence.json`, `REPRODUCE.md`.

1. Create a separate manuscript scaffold; copy only relevant background/template material with existing results clearly marked historical.
   Include the exact user-specified Qwen3 scope statement from Section 1 in the experimental setup and retain its distinction in the limitations and comparison-table captions. Do not use shared-vocabulary Qwen3 results as evidence for heterogeneous-vocabulary DFlash support.
2. Generate every new result table/plot from audited new evidence. A missing cell stays pending or N/A; it never inherits old TPS.
3. Verify related-work novelty and cite acceptance-aware training, drafter adaptation, DFlash/EAGLE, and vocabulary-transfer methods accurately.
4. Write contributions from supported findings, including negative scaling/transfer results and total cost. Separate the AUF extension's contribution from RelaySpec's original drafter-reuse contribution.
5. Compile, audit references/claims/counts, render every PDF page and inspect figures/tables for legibility; produce final PDF, raw outputs, model/config pins, and reproducible commands.

## 13. Planned paper figures and tables

Main-paper candidates, selected for distinct questions rather than repeating the same gain:

1. Method diagram: old feature alignment, AUF first-failure supervision, frozen modules, and merged deployment.
2. Main transfer table: every principal source→target pair, AR/native/ZIP/CE/AUF, speedup, accepted progress, exact agreement, quality, and fit cost.
3. Data-scaling curves from16 through32,768 records, separate fixed-update and fixed-epoch panels.
4. Compute/epoch curves showing throughput AND active-prefix learning, with training GPU-time axis.
5. Capacity/cost plot: trainable rank/parameters, deployed parameters, throughput, and fitting cost; identify dense and MLP controls.
6. Acceptance-survival and first-failure histograms explaining where AUF changes draft behavior.
7. Workload/family and batched-serving generalization panels.
8. Runtime cost decomposition and adaptation break-even plot.

Appendix: all loss/initialization/seed/block-size cells, complete benchmark quality/exact matches, additional profiles, bridge coverage, numerical diagnostics, and final Transformers replication. Full curves retain poor endpoints. Limitations discuss measured transfer restrictions, objective/exposure bias, computational cost, vocabulary support, and statistical scope honestly.

## 14. Duration and completion criteria

Do not promise a fixed short completion time before measuring the AUF trainer and cross-family integration. The two likely large costs are full-prefix feature capture and full-length repeated evaluation, not the number of trainable mapper weights alone.

After the pilot, forecast:

`GPU-hours = rollout + capture + all fit trajectories + evaluation + profiling + retries`,

with cache reuse and hardware parallelism listed explicitly. Fit time uses observed blocks/s or updates/s for each family and length bucket. Evaluation uses actual generated token counts and request latency, not a guess that every request hits the cap.

Illustrative scheduling arithmetic, not measured AUF performance:128×2048 is262,144 output tokens at the maximum cap. At hypothetical150tok/s, one method/workload/repeat needs about29minutes on one worker; at50tok/s, about87minutes, excluding setup. Four balanced request shards reduce that component toward7–22minutes, but do not remove AR, other methods, workloads, repeats, or model loading. Three runtime repeats multiply it bythree.

This is a multi-stage study likely requiring multiple days on four GPUs if the full matrix is executed, with an estimate updated from the pilot rather than an unsupported ten-minute promise. Correctness and new runtime integration also have uncertain engineering time.

The study is complete when:

- Every required transfer row is either validated and evaluated or has a concrete documented compatibility limitation; unsupported exact AUF claims are removed.
- AUF, uniform CE, and ZIP controls have matched new-runtime results and disclosed budgets.
- Data, compute, and capacity effects are separately measured through the requested ranges.
- Main conclusions survive the declared seed/repeat/confirmation checks or are written as unresolved.
- All headline rows have complete128-request evidence at2048 cap, plus original-paper breadth where specified.
- Main GPU profiling and final1–2 Transformers experiments are complete.
- The separate manuscript/PDF and reproduction artifacts are audited; no original-paper evidence is overwritten and no anticipated result is presented as observed.

Implementation starts with the generic AUF tests and Q8 compatibility pilot after this plan stage. Broad scheduling waits for those gates, not for a favorable speed result.
