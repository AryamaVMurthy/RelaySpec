# RelaySpec Part 2: Grounded EAGLE-3 Generality Plan

> **Historical design draft.** The executed, budget-pruned protocol superseding
> this draft is
> `2026-08-28-relayspec-rigorous-evidence-and-final-benchmarks.md`. In
> particular, the final DeepSpec environment pins Transformers 5.10.2, and the
> unneeded data-size/rank/concurrency sweeps below are not pending paper
> requirements.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Demonstrate that frozen-proposer interface transplantation works for a second, architecturally different proposer family by relaying the official Qwen3-4B EAGLE-3 proposer to Qwen3-8B and Qwen3-14B, with exact target verification and a claim-to-evidence-complete evaluation.

**Architecture:** Reuse the official DeepSpec Qwen3-4B EAGLE-3 proposer. Learn a target-specific linear map from five verified target hidden states to the 2,560-dimensional post-fusion context consumed by the frozen source proposer. Compare against an optimized source-reuse path, native autoregressive decoding, and official target-specific EAGLE-3 in the existing RelaySpec benchmark harness.

**Tech Stack:** Python 3.12, PyTorch 2.9, Transformers 5.3, official DeepSpec EAGLE-3, four RTX 6000 Ada GPUs, Slurm, pytest, EvalPlus, official Qwen math evaluator.

---

## 1. Evidence discipline

Every design choice in Part 2 must be one of:

1. an algebraic consequence of the frozen interface;
2. copied from a pinned official released configuration;
3. selected on a disjoint validation set by a stated metric;
4. retained from Part 1 because a complete measured ablation supports it.

No unexplained coefficient is part of the primary method. In particular, the earlier
`MSE + 0.1 * cosine` objective is not reused: `0.1` was a heuristic, and MSE already
penalizes both angular and norm error.

### Claim-to-evidence map

| Planned claim | Required evidence | Status before Part 2 |
|---|---|---|
| Interface transplantation is not DFlash-specific | EAGLE-3 4B->8B and 4B->14B source/relay comparisons | Not yet measured |
| The relay removes a material bottleneck | Component profile and Amdahl bound for optimized EAGLE source reuse | Not yet measured |
| Output correctness is preserved | Exact verifier argument, implementation equivalence tests, paired output/score audit | Speculative-decoding theorem exists; EAGLE implementation not yet audited |
| Post-fusion is the correct transplant boundary | Algebraic expressivity/parameter analysis plus pre/post landing ablation | Algebra complete; empirical ablation pending |
| Five target depths are useful | Official EAGLE-3 mechanism plus 1/3/5-tap ablation | Published evidence exists for multi-depth fusion; transfer-specific evidence pending |
| Adaptation is cheap | Measured examples, steps, GPU-minutes, adapter parameters and memory | DFlash evidence exists; EAGLE evidence pending |
| Speedup is explained rather than incidental | Predicted versus observed speedup using measured component shares and acceptance | DFlash evidence exists; EAGLE evidence pending |

## 2. External grounding

### 2.1 Exact verification

Leviathan et al., ICML 2023, prove that speculative decoding can preserve the
target distribution when the target performs the acceptance and correction step:

- https://proceedings.mlr.press/v202/leviathan23a.html

DeepSpec's evaluator implements target verification and rejection/correction; the
relay supplies proposals only:

- https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/eval/base_evaluator.py

Consequently, relay error can change verifier-call count and speed, but not the
accepted target distribution, subject to correct implementation and numerical
precision. We still score paired outputs to catch implementation bugs.

### 2.2 Why EAGLE-3 is the second family

EAGLE-3 is a peer-reviewed NeurIPS 2025 autoregressive proposer that directly
predicts tokens from fused lower/middle/upper target features. Its published
ablation improved MT-Bench speedup from 3.82x to 4.40x and GSM8K speedup from
3.77x to 4.48x when fused features replaced top-only features:

- https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html

This is mechanistically distinct from DFlash's parallel block-diffusion proposal.
Success on both families supports a claim about frozen feature interfaces rather
than one draft architecture.

### 2.3 Why the official DeepSpec release is used

DeepSpec exposes Qwen3-4B, Qwen3-8B and Qwen3-14B EAGLE-3 checkpoints under one
evaluator, eliminating reproduction ambiguity:

- https://github.com/deepseek-ai/DeepSpec

The Qwen3 implementation concatenates exactly five target states and applies
`fc: R^(5d) -> R^d`; if the caller supplies an already projected `d`-dimensional
state, the projection is bypassed:

- https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/eagle3/qwen3/modeling.py

The released Qwen3-4B configuration fixes source taps `[1, 9, 17, 25, 33]`, one
draft layer and maximum proposal length seven:

- https://github.com/deepseek-ai/DeepSpec/blob/main/config/eagle3/eagle3_qwen3_4b.py

The released target taps are `[1, 9, 17, 25, 33]` for Qwen3-8B and
`[1, 10, 19, 28, 37]` for Qwen3-14B. These are primary settings, not tuned on
our test benchmarks.

## 3. Grounded method

### 3.1 Interface definition

For source Qwen3-4B `S`, let

```
z_S(t) = concat(h_S^1(t), h_S^9(t), h_S^17(t), h_S^25(t), h_S^33(t))
c_S(t) = F_E(z_S(t))
```

where `F_E` is the frozen 4B EAGLE-3 fusion layer and `c_S(t)` has dimension
2,560. For target `T`, use its five official depth-spaced taps and learn

```
c_hat_S(t) = R_phi(N(z_T(t))).
```

`N` is evaluated as a controlled input-normalization ablation: raw input versus
per-tap RMS calibration estimated on the training split only. The winner is
selected on disjoint validation throughput. The relay output remains raw because
the frozen EAGLE layer both normalizes it internally and uses it in a residual;
forcing a unit output norm would change that source interface.

### 3.2 Why post-fusion transplantation is primary

For Qwen3-8B, a pre-fusion linear relay requires approximately

```
(5 * 4096) * (5 * 2560) = 262.14M weights,
```

whereas the post-fusion relay requires

```
(5 * 4096) * 2560 = 52.43M weights.
```

The same ratio is exactly five for Qwen3-14B: 327.68M versus 65.54M weights.
Moreover, every pre-fusion linear map `A` is observed by the proposer only through
`F_E A`. A direct post-fusion map `B` has precisely that input/output shape and can
represent the same composite map without the five-times-larger intermediate.
Post-fusion is therefore no less expressive at the consumed linear interface and
is five times smaller. The pre-fusion variant remains one empirical check.

### 3.3 Primary loss: normalized interface error

Use one coefficient-free loss:

```
L_int = mean_t ||c_hat_S(t) - c_S(t)||_2^2 / ||c_S(t)||_2^2.
```

The denominator is clamped only at the floating-point minimum positive normal
value. This is relative squared error: it is dimensionless, automatically handles
the interface's token-dependent scale, and penalizes both direction and magnitude.
Indeed, with `rho = ||c_hat|| / ||c||` and angle `theta`, each token contributes

```
rho^2 + 1 - 2 rho cos(theta).
```

Thus an additional cosine term duplicates a component already present in the
objective and would introduce an unjustified trade-off coefficient.

This feature regression trains only the translator; it is not the feature-prediction
constraint removed by EAGLE-3. EAGLE-3 itself remains frozen and continues direct
token prediction.

### 3.4 Functional refinement is an ablation, not an assumed ingredient

Part 1 provides direct evidence against automatically adding a KL term: on complete
MATH-500, the conservative joint objective changed throughput by only 1.0005x with
a confidence interval crossing one. Therefore feature-only remains the prior.

For EAGLE-3, compare two separately optimized candidates rather than choosing an
arbitrary mixture coefficient:

1. interface-only: minimize `L_int`;
2. functional refinement: initialize from interface-only and minimize the frozen
   proposer's seven-step teacher-forced KL alone.

The second candidate uses EAGLE's official step decay `0.8` only because that value
is part of the released checkpoint's training configuration. It is selected only
if paired validation throughput improves and the lower confidence bound exceeds
one. No KL coefficient is mixed into the primary loss.

### 3.5 Training-budget grounding

Use the existing 4,096-example non-thinking math manifest because it is identical
to Part 1 and disjoint from all final evaluation manifests. Do not assume that 4,096
is necessary. Train nested subsets of 512, 1,024, 2,048 and 4,096 examples for one
epoch each. These geometric sizes measure a learning curve efficiently and follow
EAGLE-3's own emphasis on data scaling.

Select the smallest data size whose paired validation throughput is statistically
indistinguishable from the 4,096-example candidate. Final claims report the whole
curve and the selected cost. The final test set is never used for selection.

Use AdamW with zero weight decay because the model is a single linear translator
and the released EAGLE-3 training configuration also uses zero weight decay. Select
the learning rate once using a short logarithmic range test on the training split;
freeze it for both target scales. Gradient clipping is retained only if the range
test observes non-finite or explosive gradients, and its bound is then derived from
the observed pre-clipping gradient-norm distribution and logged.

## 4. Amdahl-derived success condition

Do not use fixed acceptance-retention gates.

Profile optimized source reuse and write its time fractions as

```
p_S + p_C + p_O = 1,
```

where `p_S` is removable source work, `p_C` is work repeated per speculative
cycle (target verification plus drafting), and `p_O` is non-cycle overhead. Let
`p_R` be relay cost normalized by source-baseline time. If mean committed tokens
are `a_S` and `a_R`, the predicted normalized relay time is

```
T_R / T_S = (a_S / a_R) * p_C + p_O + p_R.
```

Therefore the relay is predicted faster exactly when

```
a_R / a_S > p_C / (1 - p_O - p_R).
```

For a desired speedup `s_0`, the required acceptance retention is

```
a_R / a_S >= p_C / (1 / s_0 - p_O - p_R).
```

These thresholds are computed from the live profile separately for each target,
task and proposal length. The experiment reports the predicted and observed
speedup rather than declaring a universal 90% threshold.

The ideal acceptance-preserving bound is

```
S_max = 1 / (1 - p_S + p_R).
```

The initial bottleneck probe proceeds only when its lower confidence bound implies
an ideal speedup materially above timing noise. The practical-effect threshold is
set from repeated baseline timing variance, not chosen in advance.

## 5. Controlled ablations

All ablations use one-factor-at-a-time changes on a disjoint 64-prompt validation
manifest. No Cartesian sweep is permitted.

| Mechanistic question | Variants | Grounding |
|---|---|---|
| Is the post-fusion boundary sufficient? | pre-fusion, post-fusion | algebraic five-times parameter difference |
| Is cross-model scale calibration needed? | raw, per-tap RMS | target/source families have different widths and activation scales |
| How much depth information transfers? | one late tap, low/mid/high taps, five official taps | EAGLE-3 published multi-depth ablation; DeepSpec uses five |
| How much data is needed? | 512, 1,024, 2,048, 4,096 | geometric learning curve; final selection off test set |
| Does functional alignment help EAGLE? | interface-only, KL-only refinement | Part 1 says joint loss is not automatically useful; EAGLE directly predicts tokens |
| How much relay capacity is useful? | ranks 256, 512, 1,024, full | geometric parameter-efficiency curve; full map is latency-cheap in Part 1 |
| What proposal length maximizes throughput? | 3, 5, 7 | released cap is seven; EAGLE-3 serving paper uses chain length three at high batch |
| Is training stable? | three selected-config seeds | mapper initialization is the relevant stochastic variable |

Selection metric is end-to-end output tokens/s, with accepted tokens and interface
error reported as mechanism diagnostics. A lower proxy loss cannot override worse
measured throughput.

## 6. Final experiment matrix

### Models and methods

For each of Qwen3-8B and Qwen3-14B, run:

1. native autoregressive target;
2. official target-specific EAGLE-3;
3. optimized Qwen3-4B EAGLE-3 source reuse;
4. selected EAGLE RelaySpec.

The source baseline extends a cached Qwen3-4B state with newly committed tokens
only. It never recomputes the prefix or runs the source on rejected candidates.

### Benchmarks

Reuse immutable Part 1 manifests:

| Dataset | Size | Purpose |
|---|---:|---|
| MATH-500 | 500 | primary non-thinking reasoning workload |
| GSM8K subset | 128 | short math workload |
| HumanEval | 164 | code generation |
| EvalPlus MBPP | 378 | official code robustness |
| MT-Bench | 160 turns | longer conversational workload |

No AIME and no full GSM8K. The suite spans math, code and chat because both
EAGLE-3 and DFlash evaluate across task types, and it exactly matches Part 1.

### Metrics and why they exist

| Metric | Mechanistic role |
|---|---|
| end-to-end output tokens/s | primary user-visible efficiency outcome |
| request latency, TTFT, TPOT | separates prefill and decode behavior |
| accepted tokens per verification | determines verifier-call reduction |
| per-position acceptance | detects autoregressive error accumulation |
| verifier calls/output token | direct cost mediator |
| component CUDA and wall times | identifies the removed share and validates Amdahl |
| peak/steady allocated memory | measures the benefit of unloading Qwen3-4B |
| joules/output token | ensures speed is not bought through disproportionate power |
| source/relay output agreement | implementation audit under greedy decoding |
| official task accuracy | guards against scoring or verifier bugs |
| percentage of target-specific EAGLE throughput | measures how much expensive retraining performance is recovered |
| adapter parameters, examples and GPU-minutes | quantifies portability cost |

EAGLE-3 itself uses strict acceptance and therefore treats speedup and acceptance
as primary metrics rather than expecting a quality change. We additionally score
quality because cross-model integration creates more implementation risk than the
original same-target proposer.

### Statistical protocol

- Keep the final test manifests untouched until the variant is frozen.
- Interleave source and relay runs per prompt to reduce thermal and load drift.
- Warm up every loaded method before timed records.
- Use prompt-paired bootstrap intervals for throughput ratios and acceptance deltas.
- Report point estimate plus 95% interval; emphasize effect size, not a p-value.
- Use three seeds only for the selected 8B training configuration; deterministic
  inference variation is handled by prompt-paired intervals.
- Run ablation selection only on validation. Run each final benchmark once after
  freezing the method, preventing repeated test-set tuning.

## 7. Implementation tasks

### Task 1: Pin DeepSpec and checkpoint metadata

**Files:**
- Create: `configs/deepspec_eagle3_sources.yaml`
- Modify: `docs/research/relayspec-source-log.md`
- Test: `tests/test_configs.py`

Record the DeepSpec commit, Qwen3-4B/8B/14B EAGLE checkpoint revisions, target
model revisions, tokenizer revision, official taps, hidden sizes, proposal length
and license. Add tests rejecting unpinned model identifiers.

Run:

```bash
pytest tests/test_configs.py -v
```

Expected: all configuration tests pass.

### Task 2: Add a proposer-family abstraction

**Files:**
- Create: `src/relayspec/proposers.py`
- Modify: `src/relayspec/dflash.py`
- Modify: `src/relayspec/generation.py`
- Test: `tests/test_proposers.py`

Define the minimal operations `extract_interface`, `initialize`, `propose`,
`update_after_verify` and `unload_source`. Keep target verification in the existing
generation layer so no proposer can approve output.

Run:

```bash
pytest tests/test_proposers.py tests/test_core.py -v
```

Expected: existing DFlash behavior remains unchanged.

### Task 3: Implement the official EAGLE-3 backend

**Files:**
- Create: `src/relayspec/eagle3.py`
- Modify: `src/relayspec/proposers.py`
- Test: `tests/test_eagle3.py`

Adapt DeepSpec's cache initialization, seven-step proposal, cache crop and committed
token update. Add the critical bypass test:

```python
raw = eagle(context_5d, input_ids=ids, return_logits=True)
projected = eagle(eagle.fc(context_5d), input_ids=ids, return_logits=True)
torch.testing.assert_close(raw.draft_logits, projected.draft_logits)
```

Also compare the wrapper to the official evaluator on fixed 4B prompts.

Run:

```bash
pytest tests/test_eagle3.py -v
```

Expected: raw/projected and wrapper/official paths match within pinned BF16 tolerances.

### Task 4: Replace heuristic feature loss for Part 2

**Files:**
- Modify: `src/relayspec/losses.py`
- Modify: `scripts/train_relay.py`
- Test: `tests/test_losses.py`

Add `normalized_interface_error`. Test scale invariance, exact zero, the
norm/cosine decomposition and finite handling. Preserve the Part 1 objective behind
its existing config so prior artifacts remain reproducible.

Run:

```bash
pytest tests/test_losses.py -v
```

Expected: Part 1 loss tests and new coefficient-free loss tests pass.

### Task 5: Add EAGLE training and validation configurations

**Files:**
- Create: `configs/train_eagle_relay_qwen3_8b_4gpu.yaml`
- Create: `configs/train_eagle_relay_qwen3_14b_4gpu.yaml`
- Create: `configs/eagle_ablation_manifest.json`
- Modify: `scripts/train_relay.py`
- Test: `tests/test_configs.py`

Add family, interface landing, normalization, data subset, relay rank and objective
fields. Require exactly four GPUs and forbid paths overlapping final manifests.

Run:

```bash
pytest tests/test_configs.py -v
```

Expected: all EAGLE configurations validate and leakage checks pass.

### Task 6: Extend benchmarking and component profiling

**Files:**
- Modify: `src/relayspec/benchmarking.py`
- Modify: `src/relayspec/profiling.py`
- Modify: `scripts/benchmark_relay.py`
- Test: `tests/test_benchmark.py`
- Test: `tests/test_metrics.py`

Add regions for source prefill/update, EAGLE draft steps, target verification,
relay, cache operations and unattributed time. Add dynamic Amdahl and required
acceptance calculations from Section 4.

Run:

```bash
pytest tests/test_benchmark.py tests/test_metrics.py -v
```

Expected: component shares sum to one within tolerance and synthetic observed
speed matches the analytic model.

### Task 7: Run the four-GPU baseline and bottleneck gate

**Files:**
- Create: `configs/benchmark_eagle_qwen3_8b_profile_4gpu.yaml`
- Create: `configs/benchmark_eagle_qwen3_14b_profile_4gpu.yaml`
- Create: `slurm/benchmark_eagle_profile.sbatch`
- Create after execution: `reports/eagle-profile/RESULTS.md`

Run the fixed 32-prompt profile with native AR, native target EAGLE, optimized source
reuse and official 4B wrapper checks. Compute the timing-noise-aware Amdahl bound
before training or full evaluation.

### Task 8: Run the validation ablations and freeze the method

**Files:**
- Create: `scripts/run_eagle_ablations.py`
- Create: `scripts/select_eagle_variant.py`
- Test: `tests/test_eagle_selection.py`
- Create after execution: `reports/eagle-ablations/RESULTS.md`

Run the one-factor-at-a-time matrix in Section 5. Selection code must read validation
artifacts only and emit one immutable selected configuration with its hash.

### Task 9: Run complete 8B and 14B benchmarks

**Files:**
- Create: `configs/benchmark_eagle_qwen3_8b_full_4gpu.yaml`
- Create: `configs/benchmark_eagle_qwen3_14b_full_4gpu.yaml`
- Create: `slurm/benchmark_eagle_full.sbatch`
- Create after execution: `reports/eagle-d8/RESULTS.md`
- Create after execution: `reports/eagle-d14/RESULTS.md`

Run all five frozen benchmarks, official scorers, output agreement, component
profiles, memory and telemetry. Never change the selected checkpoint afterward.

### Task 10: Add the second-family paper evidence

**Files:**
- Modify: `reports/FINAL_RESULTS.md`
- Modify: `docs/METHOD.md`
- Modify: `docs/RelaySpec_End_to_End_Research_Report.tex`
- Modify: `references.bib`
- Create: `reports/claim-evidence-map-part2.md`

Add the two-family headline table, Amdahl predicted-versus-observed plot,
acceptance-by-position plot, data/adapter Pareto curve and minimal causal ablation
table. Label EAGLE-3 as NeurIPS 2025 and DFlash as ICML 2026; cite the
camera-ready manuscript and official ICML program until the proceedings volume
is published.
Record exact proceedings URLs, and distinguish both peer-reviewed papers from
newer preprints such as DFlare and from software or technical notes.

## 8. Resource estimate

Using exactly four RTX 6000 Ada GPUs:

| Stage | Four-GPU wall time | Output |
|---|---:|---|
| integration/profile | 2-4 h | verified backend and live Amdahl bound |
| nested data fits and ablations | 4-7 h | selected evidence-grounded variant |
| complete 8B suite | 4-6 h | second-family primary result |
| complete 14B suite | 6-9 h | scale generality |
| memory/multi-request audit and scoring | 2-4 h | systems evidence |

Expected total: 18-30 four-GPU wall hours, or 72-120 GPU-hours. Stop-and-select
logic uses validation results to avoid launching full runs for inferior variants.

## 9. Paper acceptance criterion

The Part 2 result is strong when all of the following are supported by completed
artifacts:

1. optimized source reuse contains a measured removable source component;
2. the relayed EAGLE proposer eliminates that component;
3. observed speed agrees with the profile-plus-acceptance model;
4. the result is positive at 8B and 14B under the derived speed condition;
5. target verification and official scores show no quality degradation;
6. the same mechanism now works for parallel DFlash and autoregressive EAGLE-3;
7. adaptation cost is reported directly rather than compared using an estimated
   full-proposer training cost.

The defensible contribution is therefore not “a 0.1-weighted alignment loss.” It
is a measured systems principle: transplant the smallest sufficient frozen-proposer
interface, and predict the realized gain from the removable critical-path share and
the acceptance retained after translation.
