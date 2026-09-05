# RelaySpec ICLR Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish whether a small learned interface can reuse a pretrained
speculative proposer across larger compatible target models, eliminate the
source-model trunk, preserve exact target decoding, and retain most of the
speed of an expensive target-specific proposer.

**Architecture:** A frozen proposer trained for a source model receives a
learned translation of hidden states already computed by a larger verifier.
Training has a feature-reconstruction warm start followed by an
inference-matched, early-position-weighted block objective. The target always
performs exact verification; the relay affects proposal quality and cost only.

**Tech Stack:** PyTorch 2.9, Transformers, official DFlash and EAGLE-3
checkpoints, CUDA events, SGLang or vLLM Speculators for serving evaluation,
four RTX 6000 Ada GPUs, bf16, Qwen3 non-thinking models.

---

## 1. Paper thesis and claims

The paper must be framed as **cross-target proposer reuse**, not as a generic
replacement for target-specific DFlash.

The central deployment problem is:

> A trained feature-conditioned proposer exists for source model `S`, but a
> larger compatible target `T` has no trained proposer. Naive reuse requires
> running both `S` and `T` every cycle. RelaySpec adapts the proposer to `T`
> using a small interface and removes `S` from the transformer execution path.

The final paper should support five claims:

1. **Reuse:** A proposer trained for Qwen3-4B can accelerate Qwen3-8B and
   Qwen3-14B without retraining the proposer.
2. **Efficiency:** RelaySpec is faster than naive cross-target reuse because a
   sub-millisecond relay replaces the repeated 4B source trunk.
3. **Data and training efficiency:** Relay adaptation uses orders of magnitude
   less data and compute than training a new DFlash/EAGLE-3 proposer.
4. **Exactness and generality:** Greedy outputs match native target decoding,
   correct stochastic decoding preserves the target distribution, and the
   benefit appears for both a block-diffusion and an autoregressive proposer.
5. **Principled adaptation:** A survival objective derived from expected
   committed tokens directly optimizes the acceptance term in speculative
   throughput without a hand-chosen positional decay schedule.

The present 1.241x result supports only Claim 2 on four prompts. It is not yet
paper evidence for Claims 1, 3, or 4.

## 2. Main mathematical accounting

Every result row must include the terms in

\[
L=\frac{T_{\mathrm{source}}+T_{\mathrm{relay}}+
T_{\mathrm{draft}}+T_{\mathrm{verify}}+T_{\mathrm{other}}}{\tau},
\]

where only one of `T_source` and `T_relay` is nonzero and `tau` is committed
tokens per verification cycle. The paper reports both measured speedup and the
speedup predicted by this decomposition. A method is useful only when its
whole-request gain agrees with the component-level change after accounting for
acceptance loss.

The three primary methods have per-output-token latency

\[
L_{\mathrm{AR}}=T_T^{(1)},
\]

\[
L_{\mathrm{DFlash}}=
\frac{T_S+T_D+T_V(B)+T_O}{\tau_D},
\qquad
L_{\mathrm{Relay}}=
\frac{T_R+T_D+T_V(B)+T_O}{\tau_R}.
\]

Here `DFlash` means the naive cross-target reuse baseline, not the official
target-specific DFlash configuration. RelaySpec is better than naive reuse
exactly when

\[
\frac{\tau_R}{\tau_D}>
\frac{C_R}{C_D},
\]

where `C` is total cycle cost. This inequality is the central Amdahl gate.

The clean four-prompt probe currently gives

| Quantity | Native AR | Naive cross-target DFlash | RelaySpec |
|---|---:|---:|---:|
| Decode tokens/s | not yet measured | 105.77 | 131.24 |
| Milliseconds/output token | not yet measured | 9.455 | 7.620 |
| Committed tokens/cycle | 1 by definition | 7.067 | 6.095 |
| Approximate cycle cost | target one-token call | 66.815 ms | 46.440 ms |
| Conditioning component | none | 20.451 ms source trunk | 0.304 ms relay |

Thus `tau_R/tau_D = 0.8624`, while `C_R/C_D = 0.6951`; their ratio is the
observed `1.2408x` speedup. With unchanged acceptance, the measured component
costs predict `1.4317x`. The native-AR column must be measured before any
absolute acceleration claim is made.

## 3. Required model matrix

Run exactly four GPUs for every job. For batch-one latency experiments, assign
one independent request stream to each GPU and aggregate paired results.

| ID | Frozen proposer originally trained for | New verifier | Purpose |
|---|---|---|---|
| D-8 | Qwen3-4B DFlash-b16 | Qwen3-8B | Controlled pair with an official target-specific Qwen3-8B DFlash ceiling |
| D-14 | Qwen3-4B DFlash-b16 | Qwen3-14B | Headline scale-transfer pair and current positive result |
| E-8 | Qwen3-4B EAGLE-3 | Qwen3-8B | Second proposer family |
| E-14 | Qwen3-4B EAGLE-3 | Qwen3-14B | Generalization at the headline target size |

Do not add a third proposer family unless these four pairs pass. Do not claim
cross-tokenizer or cross-vocabulary generality; the initial paper is explicitly
for compatible vocabularies and chat templates.

## 4. Required methods and baselines

Run these rows for each applicable pair:

1. **Native AR target:** exact Qwen3 target, no speculation.
2. **Target-specific proposer:** official Qwen3-8B DFlash/EAGLE-3 checkpoint.
   This is the quality/speed ceiling for D-8 and E-8, not available for 14B.
3. **Naive cross-target reuse:** source-specific proposer plus the full
   Qwen3-4B source trunk plus the new target verifier.
4. **Relay-F:** current frozen linear relay trained only with normalized
   feature reconstruction.
5. **Relay-S:** feature warm start followed by the survival objective below;
   this is the intended main method.

One optional comparison is a newly trained target-specific proposer on the
same small adaptation set. Its role is to show that training a proposer from
scratch on 4K--16K examples underperforms adapting an already trained one.

## 5. Main method to train

### Stage A: feature-interface warm start

For source-proposer context `c_S` and relayed target context `c_hat_T`, optimize

\[
L_F=\operatorname{MSE}(\widehat c_T,c_S)
+0.1\,[1-\cos(\widehat c_T,c_S)].
\]

Keep the source, target, embeddings, LM head, and proposer frozen.

### Stage B: throughput-derived survival refinement

Sample inference-matched anchors and construct block-16 masked proposals. Pass
the relayed context through the frozen proposer. For target continuation token
`y_k^T`, define

\[
p_k=q_{\phi,k}(y_k^T).
\]

The differentiable expected-survival proxy is

\[
\widetilde\tau(\phi)
=1+\sum_{k=1}^{15}\prod_{j=1}^{k}p_j,
\qquad
L_S=-\log\widetilde\tau(\phi).
\]

Optimize

\[
L=L_F+\lambda_S L_S.
\]

This weighting is automatic. Its sensitivity to the negative log probability
at position `r` is

\[
\frac{\partial L_S}{\partial(-\log p_r)}
=\frac{\sum_{k=r}^{15}\prod_{j=1}^{k}p_j}
{1+\sum_{k=1}^{15}\prod_{j=1}^{k}p_j}.
\]

An early position receives more weight because it participates in every later
survival product. This replaces the fixed exponential schedule used by DFlash
with a model- and example-dependent weight derived from the quantity that
controls throughput.

Tune only `lambda_S` in `{0.1, 1.0}` on a frozen 128-prompt development set.
No larger hyperparameter sweep is permitted. Select the value with the highest
measured tokens/s, not the lowest representation loss.

## 6. Training data

Use a decontaminated target-generated subset of Open-PerfectBlend because the
latest xPress evaluation uses a broad math/code/chat mixture regenerated by the
target in non-thinking mode.

Create two fixed training scales per target:

| Split | Size | Composition | Purpose |
|---|---:|---|---|
| Small | 4,096 | 50% math, 25% code, 25% chat | Headline low-cost adaptation |
| Medium | 16,384 | same proportions | Determine whether acceptance saturates cheaply |

Use maximum training length 1,024 and sample block anchors rather than applying
the block loss at every position. Remove any training example with a matching
32-gram in the evaluation sets. Log dataset hashes and exact model-generated
responses.

Keep the existing 4K MATH-only relay only as a domain-specialized ablation. It
must not be the main checkpoint for code and chat results.

## 7. Evaluation workloads

Follow the current DFlash-family convention while keeping the suite affordable:

| Category | Benchmark | Evaluation size | Output cap | Metric |
|---|---|---:|---:|---|
| Math | MATH-500 | all 500 | 2,048 | official exact-answer accuracy, speed, acceptance |
| Math | GSM8K | fixed 200-prompt subset | 2,048 | exact-answer accuracy, speed, acceptance |
| Code | HumanEval | all 164 | 2,048 | pass@1 plus speed and acceptance |
| Code | MBPP | fixed 200-prompt subset | 2,048 | pass@1 plus speed and acceptance |
| Chat | MT-Bench | all 80 | 2,048 | token equality, speed, acceptance; judge score only once for native outputs |

AIME is intentionally excluded. All prompts use the official Qwen3 chat
template with thinking disabled. Temperature zero is the primary table.

Add one controlled length sweep using 64 fixed prompts:

- Prompt lengths: 128, 512, 2,048, 8,192 tokens.
- Output lengths: 128, 512, 2,048 tokens.
- Methods: native AR, naive reuse, Relay-S.

This sweep identifies the deployment region in which the removed source trunk
occupies enough of total latency to produce a material Amdahl-law gain.

## 8. Metrics to record for every request

### Primary system metrics

- Decode tokens/s and speedup versus native AR.
- Speedup versus naive cross-target reuse.
- End-to-end request latency including prefill.
- Time to first token.
- Inter-token latency and p50/p95 request latency.
- Mean committed tokens per cycle `tau`.
- Target verification calls per output token.

### Component measurements

- Target prefill milliseconds.
- Source prefill and source-trunk milliseconds.
- Relay prefill and relay decode milliseconds.
- Draft milliseconds.
- Target verification milliseconds.
- Logit-head, cache crop/commit, synchronization, and unattributed runtime.
- Peak allocated and reserved GPU memory.
- Adapter parameter count, checkpoint bytes, training wall time, and GPU-hours.

### Correctness and quality

- Native-target token-hash equality for every greedy sample.
- Official math exact-match and code pass@1.
- Stop reason and output-length cap-hit rate.
- For temperature-one sampling, distributional tests against native target
  sampling after implementing correct rejection sampling: token-frequency
  total variation and sequence-level task-score confidence intervals.

Use two warm-up runs and five timed repetitions for microbenchmarks. Alternate
method order within each paired prompt. Report paired-bootstrap 95% confidence
intervals over prompts and repetitions. Never compare numbers collected under
different attention backends, precision, maximum lengths, or GPU clocks.

## 9. Serving experiment

After batch-one gates pass, integrate RelaySpec into the same serving runtime
used for the DFlash baseline. Use SGLang if its DFlash path exposes the needed
feature interface; otherwise use vLLM Speculators for both methods.

Run global concurrency `{1, 4, 8, 16}` on the same four-GPU node for MATH-500
and HumanEval. Report:

- Aggregate output tokens/s.
- p50 and p95 latency.
- GPU memory and utilization.
- Acceptance length.
- Speedup versus AR and naive reuse.

Do not compare the current Python loop against an optimized serving baseline.

## 10. Minimal ablations

Only run ablations that test a paper claim:

1. **Objective:** feature reconstruction, uniform block CE, DFlash-style fixed
   exponential CE, and the proposed survival loss. This establishes that the
   formula, not merely extra optimization, recovers acceptance.
2. **Data:** 1K, 4K, and 16K mixed examples. Run three training seeds only for
   the final 4K Relay-S result on D-8 and D-14.
3. **Target features:** final layer only, three depth-spaced taps, and five
   depth-spaced taps.
4. **Relay capacity:** linear projection versus a parameter-matched two-layer
   residual adapter. The linear method remains main unless the residual gives
   a material whole-request gain.
5. **Speculative block:** block sizes 8, 16, and 32, reporting both cycle cost
   and acceptance rather than selecting by acceptance alone.
6. **System removal:** native AR, full source trunk, relay while retaining the
   full source model in memory, and relay loading only embedding/LM-head
   weights. This separates latency and memory contributions.
7. **Scale and proposer:** D-8, D-14, E-8, and E-14.
8. **Regime:** prompt/output-length sweep and serving concurrency.
9. **Exactness:** greedy token equality and correct temperature-one sampling.

Do not run broad projector architecture, optimizer, random-seed, or layer-cut
sweeps unless one of these ablations fails its decision gate.

## 11. Decision gates

### Gate A: measurement and exactness

Use 64 MATH-500 prompts on D-8.

- Native AR and every exact method produce identical greedy token hashes.
- Repeated native throughput varies by less than 3%.
- Profiled regions explain at least 95% of request time.

### Gate B: controlled 8B result

Using Relay-S on D-8:

- At least 1.15x throughput over naive cross-target reuse.
- At least 1.8x throughput over native AR.
- At least 90% of naive-reuse acceptance length.
- At least 80% of the official target-specific DFlash speedup.
- 100% greedy token equality.

### Gate C: headline 14B result

On 64 prompts balanced across math, code, and chat:

- At least 1.20x over naive reuse.
- At least 1.8x over native AR.
- Positive paired speedup on every category.
- Relay time below 1 ms/cycle and 100% greedy equality.

### Gate D: proposer generality

E-8 and E-14 must each show at least 1.10x over their naive source-trunk reuse
baseline without reducing exactness. If this fails, frame the method explicitly
as block-diffusion proposer reuse and do not claim proposer generality.

### Gate E: ICLR-scale evidence

Proceed to the final paper tables only if the lower bound of the 95% confidence
interval for RelaySpec versus naive reuse exceeds 1.0 on every full benchmark,
the geometric-mean gain is at least 1.15x, and the training-data/compute cost is
below 5% of training a target-specific proposer.

## 12. Theory and proof obligations

The paper needs formal statements, not only empirical equality.

### Theorem 1: exact greedy output

Prove by induction over verification cycles. The invariant is that the
committed prefix and target KV cache equal native greedy decoding. For an
arbitrary proposed block, every token before the first mismatch has the same
conditioning prefix as native decoding; the verifier correction at the first
mismatch is therefore the native next token. Cropping rejected KV entries
restores the invariant. The relay may be arbitrarily inaccurate without
changing the conclusion.

### Theorem 2: exact stochastic output

After implementing rejection sampling, accept a proposal token `x` with

\[
a(x)=\min\{1,p(x)/q(x)\}.
\]

On rejection, sample from normalized `(p-q)_+`. Show for every token `x` that

\[
q(x)a(x)+
\left(1-\sum_v\min\{p(v),q(v)\}\right)
\frac{(p(x)-q(x))_+}{\sum_v(p(v)-q(v))_+}
=p(x).
\]

Apply the identity sequentially to the block. The proof must use the actual
proposal probabilities emitted by the DFlash/EAGLE-3 implementation.

### Proposition 1: throughput condition

From the renewal-reward identity `L=C/tau`, derive

\[
S_{R/D}=\frac{L_D}{L_R}
=\frac{\tau_R/\tau_D}{C_R/C_D}.
\]

This proves the relay is beneficial exactly when relative acceptance retained
exceeds relative cycle cost retained. Verify this equality numerically for
every benchmark and show predicted versus measured speedup.

### Proposition 2: survival objective

Let `M_k` denote matching the target at proposal position `k`. Using the tail
sum identity,

\[
\mathbb E[\tau]=1+\sum_{k=1}^{15}
\Pr(M_1\cap\cdots\cap M_k).
\]

Replacing hard match events by proposer target-token probabilities gives
`tau_tilde`. Differentiate it to obtain the automatic positional weights in
Section 5. This is the derivation of the main training formula.

### Complexity statement

Report exactly what disappears: the source transformer's layer FLOPs and
source KV-cache growth. Report exactly what remains: target verification,
drafting, target hidden taps, token embedding/LM head, and the 65.5M-parameter
relay. Do not describe the method as literal target-KV transfer.

## 13. Paper tables and figures

The final paper needs exactly these main artifacts:

1. **Table 1:** D-8 and D-14 across five benchmarks: AR, naive reuse, Relay-F,
   Relay-S, and target-specific ceiling where available.
2. **Table 2:** E-8/E-14 generalization with speedup and acceptance.
3. **Table 3:** Adaptation data, trainable parameters, wall time, GPU-hours,
   peak memory, and achieved speedup versus target-specific training.
4. **Table 4:** Serving concurrency throughput and p95 latency.
5. **Figure 1:** End-to-end inference diagram showing the removed source trunk.
6. **Figure 2:** Amdahl breakdown per output token for naive reuse and RelaySpec.
7. **Figure 3:** Speedup versus acceptance/data scale Pareto plot.
8. **Figure 4:** Prompt/output length regime heatmap.
9. **Figure 5:** Predicted versus measured speedup from
   `(tau_R/tau_D)/(C_R/C_D)` with confidence intervals.

## 14. Execution order and estimated four-GPU cost

### Phase 1: fast gates, approximately 4--6 wall hours

1. Add native AR equality and complete component accounting.
2. Port D-8 and run Gate A.
3. Train 4K mixed Relay-F and Relay-S for D-8.
4. Run Gate B.
5. Apply Relay-S to D-14 and run Gate C.

Expected budget: 16--24 GPU-hours.

### Phase 2: core paper results, approximately 12--18 wall hours

1. Generate/decontaminate the 16K mixed training set.
2. Train final D-8 and D-14 relays.
3. Run the full five-benchmark greedy suite.
4. Run the length-regime sweep.
5. Produce Tables 1 and 3 and Figures 2--4.

Expected cumulative budget: 64--96 GPU-hours.

### Phase 3: generality and serving, approximately 12--20 wall hours

1. Implement the EAGLE-3 interface adapter and run Gate D.
2. Implement correct temperature-one rejection sampling and validate it.
3. Integrate the passing methods into one serving runtime.
4. Run the concurrency suite and assemble Tables 2 and 4.

Expected total budget: approximately 120--176 GPU-hours, or 30--44 wall hours
on the fixed four-GPU node. Stop at a failed gate rather than spending the full
budget on a weak claim.

## 15. Implementation task map

### Task 1: Freeze experiment contracts

**Files:**

- Create: `src/relayspec/config.py`
- Create: `src/relayspec/metrics.py`
- Create: `tests/test_metrics.py`
- Modify: `configs/eval_manifest.json`

Add typed contracts for method identity, model revisions, prompt identity,
generation settings, CUDA timing regions, and result schema. Test that rows
with mismatched backends or generation caps cannot be aggregated.

### Task 2: Add native-target equality baseline

**Files:**

- Modify: `src/relayspec/generation.py`
- Modify: `scripts/benchmark_relay.py`
- Create: `tests/test_exact_greedy.py`

Compare every method directly with native target greedy tokens. Add cache-crop
tests in which mismatches occur at positions 1, middle, and end of a block.

### Task 3: Implement acceptance-aware relay training

**Files:**

- Modify: `scripts/train_relay.py`
- Create: `src/relayspec/losses.py`
- Create: `tests/test_losses.py`
- Create: `configs/train_relay_acceptance_qwen3_8b_4gpu.yaml`
- Create: `configs/train_relay_acceptance_qwen3_14b_4gpu.yaml`

Add random inference-matched block anchors, stable log-space cumulative
products, and the survival loss. Test its closed-form value and gradient
against enumeration on blocks of length 2--4. Verify that proposer, source,
and target gradients remain absent and only relay parameters change.

### Task 4: Build the paired benchmark runner

**Files:**

- Modify: `scripts/benchmark_relay.py`
- Create: `scripts/aggregate_results.py`
- Create: `tests/test_aggregation.py`
- Create: `configs/benchmark_core_qwen3_8b_4gpu.yaml`
- Create: `configs/benchmark_core_qwen3_14b_4gpu.yaml`

Add alternating method order, five repetitions, native AR, source reuse,
Relay-F, Relay-S, and target-specific ceiling. Emit per-prompt raw JSONL and a
summary with paired-bootstrap confidence intervals.

### Task 5: Execute Gates A--C

**Files:**

- Create: `slurm/train_relay_acceptance.sbatch`
- Create: `slurm/benchmark_core.sbatch`
- Create: `reports/gates/README.md`

Every job must assert exactly four visible GPUs, record allocation metadata,
save stdout/stderr/config/model hashes, and write the gate decision without
deleting raw results.

### Task 6: Full evaluation and official scoring

**Files:**

- Create: `src/relayspec/evaluation.py`
- Create: `scripts/score_outputs.py`
- Create: `tests/test_evaluation.py`
- Create: `configs/benchmark_full_4gpu.yaml`

Implement official math answer extraction and invoke EvalPlus-compatible code
scoring in a separately sandboxed process. Verify that all exact methods share
the same scores because their emitted tokens match.

### Task 7: Second proposer interface

**Files:**

- Create: `src/relayspec/interfaces/base.py`
- Create: `src/relayspec/interfaces/dflash.py`
- Create: `src/relayspec/interfaces/eagle3.py`
- Create: `tests/test_interfaces.py`

Define the minimal proposer contract: extract source conditioning, translate
target features, draft, and commit/crop cache. Keep each proposer-specific
relay independently trained; do not require one universal set of weights.

### Task 8: Serving and final reporting

**Files:**

- Create: `scripts/benchmark_serving.py`
- Create: `scripts/build_paper_tables.py`
- Create: `tests/test_paper_tables.py`
- Create: `configs/benchmark_serving_4gpu.yaml`

Run concurrency `{1,4,8,16}` and generate all paper tables directly from raw,
immutable result files. Every table cell must link back to job ID, config hash,
model revision, and prompt manifest.

## 16. Immediate next command sequence

Do not start with the full benchmark. Implement Tasks 1--3, then execute only
Gate A and Gate B on Qwen3-8B. The first decisive result is whether
acceptance-aware RelaySpec can recover at least 90% of the source-conditioned
acceptance while retaining a 1.15x whole-request gain over naive reuse.
