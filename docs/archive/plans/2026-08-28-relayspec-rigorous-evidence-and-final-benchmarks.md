# RelaySpec Rigorous Mechanism and Final Benchmark Plan

> **Execution note:** This protocol freezes the scientific decisions before new
> full benchmark runs. It covers both DFlash and EAGLE-3, uses Qwen3
> non-thinking models, excludes AIME, and never requests more than four GPUs.

> **Completion record (2026-08-28):** Tasks 1--10 are complete. All GPU work
> used exactly four RTX 6000 Ada GPUs and was serialized. The authoritative
> final matrix is `reports/final/BREADTH_MATRIX.{json,md}`; the evidence ledger
> is `reports/FINAL_RESULTS.md`; remote four-GPU verification is in
> `reports/final/remote-4gpu-verification/`; and the rendered report is
> `output/pdf/RelaySpec_End_to_End_Research_Report.pdf`. Conceptual filenames
> below were implementation targets, not extra required duplicates: exactness
> coverage landed in `tests/test_benchmark.py`, `tests/test_eagle3.py`, and
> provider-contract tests; profiling artifacts landed in
> `reports/optimized-source-profile/` and
> `reports/eagle3-optimized-source-profile/`; final train/validation configs
> are the audited files under `configs/protocol_active/`; large checkpoints
> remain in their immutable cluster artifact directories while hashes and
> config snapshots are stored locally. The corrected EAGLE two-turn chat runs
> supersede the incomplete first-turn chat rows.

**Goal:** Establish whether a small target-specific interface translator can
reuse a frozen proposer trained for Qwen3-4B when the verifier is Qwen3-8B or
Qwen3-14B, while preserving exact target-model decoding and delivering measured
end-to-end speed and memory gains for two distinct proposer families.

**Primary scientific claim:** RelaySpec is *interface transplantation*, not a
new proposer. It replaces the repeated source-model conditioning trunk of an
already-trained feature-conditioned proposer with a translator from hidden
states that the target verifier already computes. The target verifier remains
the sole authority on committed tokens.

**Primary systems claim:** RelaySpec is useful exactly when the eliminated
source-conditioning work is a material fraction of source-reuse latency and the
translator does not reduce accepted tokens enough to erase that saving. This is
predicted before full evaluation with a measured Amdahl model and then tested
against end-to-end timings.

**Implementation strategy:** Put DFlash and EAGLE-3 behind one proposer
interface, expose the exact tensor consumed by each frozen proposer, train only
one target-specific translator, and compare native AR, official target-specific
proposer, optimized source reuse, and RelaySpec under identical verifier,
prompt, decoding, and timing code.

---

## 1. The mechanism that must work in both families

Let the frozen source model be \(S\), the target/verifier be \(T\), and a
proposer trained with \(S\) be \(P_S\). At verification cycle \(j\), the target
has already produced hidden states \(H^T_j\) and its exact KV cache. The original
source-reuse pipeline separately runs \(S\) to produce the proposer context

\[
    c^S_j = I_S(H^S_j),
\]

where \(I_S\) is the frozen family-specific conditioning interface. RelaySpec
instead computes

\[
    \widehat c^S_j = R_{T\rightarrow S}(J_T(H^T_j)),
\]

where \(J_T\) selects target features and \(R_{T\rightarrow S}\) is the only
trainable module. The same frozen proposer then emits candidates

\[
    y^{\mathrm{draft}}_j = P_S(\widehat c^S_j,\;K^P_j).
\]

The target verifies those candidates with the same algorithm used by the
official proposer implementation. RelaySpec never commits an unverified token.

### 1.1 Family-specific interface boundary

The abstraction is shared, but the boundary is not guessed.

| Family | Context that source reuse computes | Context RelaySpec must predict | Proposal structure |
|---|---|---|---|
| DFlash | the exact fused/normalized source feature tensor consumed by the frozen diffusion drafter | that post-fusion consumed tensor, with the same dtype, shape, position convention, and cache semantics | one parallel block, official block length |
| EAGLE-3 (pinned DeepSpec evaluator) | the exact output of the frozen five-depth fusion projection immediately before the frozen draft layer | that post-fusion EAGLE context, not arbitrary raw hidden states | length-7 autoregressive-chain draft steps from the released evaluator; no dynamic-tree claim |

Predicting the *post-fusion* interface is intentional. If the frozen proposer
first applies a linear fusion \(F\) to concatenated source features, any linear
pre-fusion translator \(A\) induces \(FA\). A direct post-fusion map \(B=FA\)
can express the same consumed linear function with roughly one fifth the output
width for five taps. This is an algebraic decision, then verified empirically
against pre-fusion mapping on the validation set.

### 1.2 One provider abstraction, no baseline drift

Implement exactly two context providers:

```text
SourceTrunkProvider(prefix_state) -> exact proposer context c_source
RelayProvider(target_hidden_state) -> predicted context c_relay
```

Both feed the same `ProposerBackend.initialize/propose/update` code. Native
target-specific DFlash/EAGLE use their released backend path. All methods use the
same verifier code, sampler, tokenizer, prompt template, stopping logic, CUDA
settings, and timing regions. This design makes the claimed intervention a
single replaceable component rather than a rewritten fast path.

### 1.3 Exactness and finite-precision statement

The translator is approximate; the final decoding algorithm is exact with
respect to the target under the following assumptions:

1. every committed candidate is accepted only by the target verifier;
2. rejection and correction follow the canonical speculative-decoding rule for
   the requested sampling distribution;
3. target logits, tokenization, position IDs, attention mask, KV updates, and
   stopping rules are identical to native target decoding;
4. no approximate early acceptance, tolerance-based token acceptance, or
   proposer-only termination is used.

In exact arithmetic, greedy decoding follows the same target argmax by induction
on committed tokens. For stochastic decoding, distributional exactness follows
from the standard rejection/correction argument, but the paper will claim it
only after an implementation test against the released backend. Approximate
proposer context can lower acceptance and speed; it cannot change the target
distribution under these assumptions.

Bitwise sequence identity is a separate *finite-precision implementation*
property, not part of that theorem. Different accepted-block shapes can select
different BF16 GEMM kernels and perturb near-tied logits. Therefore the paper
reports (i) byte/token agreement, (ii) first-divergence logit margins, and (iii)
official task-score deltas. It does not relabel a BF16 mismatch as a theorem
violation or silently omit it. Strict bitwise reproducibility is an optional
FP32/deterministic-kernel diagnostic, not the primary performance setting.

---

## 2. Evidence hierarchy: “backing for the backing”

Every paper statement must appear in `reports/claim-evidence-map.md` as

```text
claim -> decision -> primary evidence/derivation -> executable check
      -> immutable raw artifact -> paper table/figure
```

### 2.1 Admissible evidence classes

| Code | Evidence | Permitted use |
|---|---|---|
| F | formal derivation with explicit assumptions | exactness, algebraic equivalence, cost threshold |
| P | peer-reviewed primary paper/proceedings | published algorithm and reported experimental precedent |
| O | pinned official repository/checkpoint/config | exact released behavior, shapes, defaults, checkpoint availability |
| V | our disjoint validation experiment | selection among alternatives not fixed by F/P/O |
| T | our untouched test artifact | final scientific result only |
| A | administrative/reproducibility choice | seeds, paths, job IDs; never presented as an optimized scientific constant |

Secondary articles, blogs, and technical notes may identify related work, but
cannot be the sole support for an algorithmic or quantitative claim. A paper's
citation to another paper is not evidence: follow it to the original source.
Official code behavior is pinned by commit hash and tested locally. Our results
are supported by raw JSONL, configuration snapshots, source hashes, environment,
GPU telemetry, and scorer revisions.

### 2.2 Primary literature ledger

| Source | Status | What it supports | What it does **not** support |
|---|---|---|---|
| DFlash, accepted at ICML 2026 | peer reviewed; camera-ready manuscript and official ICML program checked | Qwen3 non-thinking setup, DFlash architecture, block-16 precedent, tasks, max 2,048 output tokens, speed/acceptance/concurrency reporting | RelaySpec translator design or its gains; an unpublished proceedings-volume number |
| EAGLE-3, NeurIPS 2025 | peer reviewed | direct-token EAGLE, low/mid/high feature value, exact verification, acceptance and serving metrics | cross-target interface transplantation |
| Speculative Decoding, ICML 2023 | peer reviewed | target-distribution exactness of canonical speculative sampling | correctness of our cache implementation |
| DeepSpec EAGLE-3 repository | official software, pin commit | actual Qwen3 4B/8B/14B checkpoints, five taps, fusion dimensions, draft cap, released configs | scientific superiority without our measurements |
| DFlare, 2026 | preprint | nearest DFlash conditioning/feature-fusion novelty threat | peer-reviewed validation of RelaySpec |
| AngelSpec, 2026 | preprint | task/load-dependent proposer and verification co-design; serving-throughput reporting | evidence for frozen-proposer transplantation or batch-one-to-serving extrapolation |
| RepSpec, ICLR 2026 | peer reviewed | training-only re-parameterization, accepted-length and inference-cost accounting | cross-target interface transplantation |
| SPEED-Bench, ICML 2026 | peer reviewed | domain diversity, context/concurrency and production-engine measurement; strict latency/throughput distinction | proof of RelaySpec or permission to extrapolate batch-one results to serving |
| SpecForge, 2026 | software/preprint | current EAGLE training/serving compatibility | evidence for our measured performance |
| SpecVocab, Findings ACL 2026 | peer reviewed | EAGLE vocabulary projection can be a nontrivial draft cost | source-trunk share on our hardware |
| Speculative KV Coding, 2026 | technical note | novelty-neighbor audit only | basis for performance or correctness claims |

For every cited numeric value, record page/table/equation or the pinned config
line. `references.bib` must contain only sources actually checked.

---

## 3. Constant and decision registry

No scientific constant enters a final config unless it has one of F/P/O/V as
its origin. The registry is machine-readable in
`configs/relayspec_protocol.yaml` and human-readable in
`docs/research/relayspec-decision-register.md`.

### 3.1 Constants already fixed without tuning

| Decision | Value | Origin and reason |
|---|---:|---|
| model family | Qwen3 non-thinking | scope fixed by study; matches available DFlash and EAGLE-3 families |
| source | Qwen3-4B | common released source proposer size |
| targets | Qwen3-8B, Qwen3-14B | two target scales with released compatible proposer baselines |
| target authority | exact verifier only | F: required for target-distribution preservation |
| DFlash output cap | 2,048 | P: DFlash main evaluation precedent; avoids truncating math/code reasoning |
| DFlash proposal block | 16 | P/O and Part-1 V: released default and measured 8/16/32 throughput winner |
| EAGLE maximum draft length | 7 | O: pinned released `ttt7` checkpoint contract and official DeepSpec path |
| target taps | released five positions | O: immutable checkpoint contract; RelaySpec transplants rather than redesigns this interface |
| temperature for primary result | 0 | P and deterministic exact-agreement test; Qwen3 remains non-thinking |
| AIME | excluded | study constraint; no result or selection uses it |
| hardware cap | exactly four GPUs | A: resource constraint, not an algorithmic claim |

### 3.2 Constants that are removed or re-selected

| Historical choice | Problem | New rule |
|---|---|---|
| `MSE + 0.1*cosine` | `0.1` lacks derivation; normalized MSE already contains angular and radial error | primary candidate is coefficient-free relative interface MSE; compare old objective only as a historical ablation |
| `feature + 0.1*KL` | arbitrary tradeoff; Part 1 full MATH gave 1.0005x and CI crossed 1 | remove from core; allow a separate KL-only refinement only if it beats interface-only on validation |
| learning rate `2e-4` | historical | matched 128-step pilots selected `6e-4` (0.413 versus 0.586 relative loss), then froze it for both targets |
| weight decay `0.01` | no reason to regularize a single linear map this way | use zero by default, matching released EAGLE config; test nonzero only if validation shows overfit |
| gradient clip `1.0` | previously under-justified | retain because it is the pinned official EAGLE training value; report it as implementation inheritance rather than an optimized RelaySpec constant |
| one warmup repetition | administrative timing setup | execute one full method warmup before measurement, rotate measured method order per request, and never count warmups; this is not claimed optimal |
| repeated full requests | would multiply the dominant evaluation cost and make thermal dependence harder to balance | execute every full-suite request once per paired method after one excluded warm-up; rotate method order and estimate uncertainty by paired whole-request bootstrap over 128--500 prompts (80 conversation clusters for MT-Bench). Development microprobes remain explicitly labeled and never replace full-suite uncertainty |
| fixed `0.1` cosine acceptance/drift threshold | scale-dependent and ungrounded | never use a quality threshold to commit tokens; report continuous normalized error and use Amdahl break-even for selection |

### 3.3 Coefficient-free primary alignment loss

For proposer-consumed context \(c_t\) and relay output \(\hat c_t\), use

\[
L_{\mathrm{rel}} = \frac{1}{N}\sum_t
\frac{\|\hat c_t-c_t\|_2^2}
     {\max(\|c_t\|_2^2,\epsilon_{\mathrm{dtype}})}.
\]

The denominator clamp is the smallest safe positive value for the accumulation
dtype, not a tuned semantic threshold. Writing
\(\rho=\|\hat c\|/\|c\|\) and \(\theta\) for the angle gives

\[
\frac{\|\hat c-c\|^2}{\|c\|^2}=\rho^2+1-2\rho\cos\theta.
\]

Thus the loss already penalizes both magnitude error and directional error. A
separate cosine coefficient duplicates part of the same geometry. Validation
still compares raw MSE, relative MSE, and the historical objective so the
argument is algebraic *and* empirical.

### 3.4 Hyperparameter selection without benchmark leakage

Use three disjoint partitions:

1. **fit:** one fixed 4,096-example MATH-training subset;
2. **development:** 32 registered MATH-500 prompts for interface/timing
   selection;
3. **independent test:** the normalized-hash 468-prompt MATH complement plus
   GSM8K-128, HumanEval, EvalPlus MBPP, and MT-Bench; never used to select
   checkpoints, interface scaling, loss, or optimizer. The conventional
   all-500 MATH table is retained for benchmark comparability and explicitly
   labeled as containing the 32 development prompts.

Hash normalized prompts and reject exact overlaps while retaining mathematical
operators. The 4,096-example count is a declared compute budget, not a claimed
minimum or optimum. A data-size sweep is deliberately omitted: the complete
fit already takes under two minutes on four GPUs and uses 130--195 times fewer
examples than the related target-specific proposer training sets, so reducing
fit data cannot materially strengthen the inference-speed claim.

---

## 4. Cost model and decision gate

Let the optimized source-reuse latency per committed output token decompose as

\[
L_S = p_C+p_S+p_O=1,
\]

where \(p_C\) is proposer plus target-verification cycle work, \(p_S\) is the
source conditioning trunk removed by RelaySpec, and \(p_O\) is other runtime.
Let RelaySpec add translator fraction \(p_R\), and let \(a_S,a_R\) be mean
committed tokens per verification cycle. A first-order prediction is

\[
\frac{L_R}{L_S}=\frac{a_S}{a_R}p_C+p_O+p_R.
\]

RelaySpec is predicted faster exactly when

\[
\frac{a_R}{a_S}>\frac{p_C}{1-p_O-p_R}.
\]

The ideal equal-acceptance speedup is

\[
s_{\max}=\frac{1}{1-p_S+p_R}.
\]

These equations replace arbitrary rules such as “cosine error below 0.1.” The
32-prompt live profile measures \(p_C,p_S,p_O,p_R,a_S,a_R\), predicts the
speedup and uncertainty, and is compared with measured end-to-end speed. A
candidate reaches full tests only when the lower confidence bound of predicted
speedup exceeds 1 and the implementation exactness suite passes.

The gate is a resource-allocation rule, not selective reporting: every tested
candidate and the reason it did or did not advance remains in the ablation
artifact.

---

## 5. Exact experiment matrix

### 5.1 Methods

Run the following for each target whenever the official checkpoint exists:

1. **Native AR:** target model alone.
2. **Native proposer:** official target-specific DFlash or EAGLE-3.
3. **Source reuse:** frozen Qwen3-4B proposer plus its separately executed 4B
   conditioning trunk; optimized and measured in the same harness.
4. **RelaySpec:** same frozen 4B proposer; source trunk replaced by selected
   target-to-source interface translator.

The central comparison is RelaySpec versus source reuse. Native AR measures
absolute usefulness. Native target-specific proposer measures the performance
recovered without retraining a full proposer.

### 5.2 Models and proposer families

| Family | Source proposer | Targets | Primary family question |
|---|---|---|---|
| DFlash | Qwen3-4B DFlash | Qwen3-8B, Qwen3-14B | does transplantation work for parallel block-diffusion proposals? |
| EAGLE-3 | Qwen3-4B EAGLE-3 | Qwen3-8B, Qwen3-14B | does the same interface idea work for sequential/tree autoregressive proposals? |

### 5.3 Evaluation workloads

| Workload | Count | Role | Quality metric |
|---|---:|---|---|
| MATH-500 | 500 | primary long non-thinking reasoning workload | official exact-answer accuracy |
| GSM8K fixed subset | 128 | short math confirmation only | exact match with frozen manifest |
| HumanEval | 164 | code generation | pass@1 with official/EvalPlus execution |
| EvalPlus MBPP | official 378 base subset already used in Part 1 | code generality | base and plus pass@1 |
| MT-Bench | 80 conversations/160 turns | chat and variable-length behavior | speed, acceptance, and paired output agreement; judge score only if the official judge pipeline is separately executed |

AIME is not run. All prompts use the appropriate official Qwen3 non-thinking
template. The 2,048-token cap is retained for comparability; emitted-length and
truncation rates are reported so throughput cannot be inflated by shorter or
prematurely terminated outputs.

### 5.4 Primary metrics

For every prompt and method record:

- wall-clock end-to-end latency, decode time, and output tokens/s; TTFT is
  unavailable because the inherited backends do not expose a synchronized
  first-token event, and no placeholder value is reported;
- prompt tokens, emitted tokens, stop reason, truncation;
- proposed, accepted, rejected, and committed tokens per cycle;
- mean accepted length and survival/acceptance at every draft position;
- target verifier cycles and target forward-pass tokens;
- component CUDA time: target forward/verification, source trunk, translator,
  proposer core, LM/vocabulary head, cache update, and other runtime;
- peak allocated/reserved GPU memory and resident parameter bytes;
- task score, token agreement, and first-divergence target-logit margin versus
  native AR for greedy decoding;
- model/checkpoint/config/code/dataset/scorer hashes.

Report the deployment-relevant aggregate throughput ratio
`(sum candidate tokens / sum candidate time) / (sum reference tokens / sum
reference time)` with a paired request-cluster 95% bootstrap CI. When paired
outputs contain exactly the same token counts this reduces to the ratio of
summed reference and candidate times. Also report the median per-request ratio,
tail latency, task-score difference with an appropriate paired CI, and
per-prompt scatter. The fixed 10,000 bootstrap draws and seed are administrative
precision choices declared before final scoring, not tuned scientific
constants.

### 5.5 Serving scope

The present implementation reports batch-one request micro-rates, p50/p95
latency, and memory with four independent ranks. SPEED-Bench demonstrates that
production throughput depends on engine, scheduling, concurrency, and batching;
therefore no concurrency sweep is mixed into this custom dense-SDPA harness and
no serving-throughput claim is made. A production-engine integration is a
separate systems study, not an unfinished requirement for the causal
source-versus-relay result.

---

## 6. Minimal causal ablations

All selection ablations run on Qwen3-8B development prompts first. Transfer only
the winner to 14B. This keeps cost controlled.

| Question | Candidates | Selection evidence |
|---|---|---|
| Which interface? | pre-fusion vs exact post-fusion consumed context | algebraic parameter count plus measured validation speed/acceptance |
| Which loss? | raw MSE, relative MSE, historical MSE+0.1 cosine | validation throughput, acceptance-by-position, interface error |
| How many target taps? | released five positions | immutable consumed-interface contract; changing them would redesign the proposer rather than transplant it |
| Does low rank help? | full rank retained | relay arithmetic is already below 0.8% of request time, so rank reduction has negligible Amdahl upside but can lower acceptance |
| How much data? | fixed 4,096-example budget | complete fit is under two minutes and 130--195x smaller than related proposer training; no minimum-data claim |
| Does functional refinement help? | none vs separate KL-only refinement | advance only if paired validation throughput CI is above 1 |
| Proposal length | DFlash 8/16/32 historical confirmation; EAGLE 3/5/7 | absolute end-to-end throughput, not acceptance alone |
| Input scaling | raw vs per-tap RMS normalization | validation stability and throughput; output remains exact interface scale |

This is the complete core ablation set. No combinatorial grid is run. The only
empirically selected choices are interface scaling, loss, and the inherited
DFlash proposal length; architecture-forced and Amdahl-dominated alternatives
are justified rather than expensively swept.

---

## 7. Execution tasks with files, tests, and commands

### Task 1: Freeze the evidence and decision register

**Create:**

- `docs/research/relayspec-decision-register.md`
- `reports/claim-evidence-map.md`
- `configs/relayspec_protocol.yaml`
- `scripts/audit_research_constants.py`
- `tests/test_research_protocol.py`

**Modify:**

- `docs/research/relayspec-source-log.md`
- `references.bib`
- `docs/METHOD.md`
- `docs/RelaySpec_End_to_End_Research_Report.tex`

**Work:**

1. Extract every numeric literal from scientific configs, method text, and
   tables.
2. Assign evidence class F/P/O/V/T/A, source locator, scope, and frozen status.
3. Record DFlash as accepted at ICML 2026, backed by the camera-ready arXiv
   record and official ICML 2026 program; do not invent a PMLR volume before
   proceedings publication.
4. Remove `0.1*cosine` and `0.1*KL` from the proposed core method; label existing
   Part-1 runs as historical ablations.
5. Add source-status labels and exact page/table/config-line locators.

**Verify:**

```bash
python scripts/audit_research_constants.py \
  --protocol configs/relayspec_protocol.yaml \
  configs/protocol_active/*.yaml
pytest -q tests/test_research_protocol.py
```

Pass condition: no unregistered scientific constant and no unsupported paper
claim.

### Task 2: Build one backend contract for DFlash and EAGLE-3

**Create:**

- `src/relayspec/proposers.py`
- `src/relayspec/eagle3.py`
- `tests/test_proposer_contract.py`
- `tests/test_eagle3.py`

**Modify:**

- `src/relayspec/dflash.py`
- `src/relayspec/generation.py`
- `src/relayspec/cache.py`
- `src/relayspec/profiling.py`

**Work:**

1. Define `ProposerBackend` and `ContextProvider` protocols.
2. Wrap the existing DFlash path without changing its numerical behavior.
3. Vendor or depend on a pinned official DeepSpec revision; implement the
   Qwen3 EAGLE-3 wrapper directly against its released tensor contract.
4. Expose source-trunk and post-fusion context hooks in both families.
5. Ensure source reuse and relay differ only by context provider.

**Verify:**

```bash
pytest -q tests/test_proposer_contract.py tests/test_eagle3.py tests/test_relay.py
```

Pass condition: DFlash regression tensors are unchanged; EAGLE native wrapper
matches official logits/proposals/cache transitions on fixed fixtures.

### Task 3: Implement coefficient-free alignment and cache exactness

**Create:**

- `src/relayspec/cost_model.py`
- `tests/test_exactness.py`
- `tests/test_cost_model.py`

**Modify:**

- `src/relayspec/losses.py`
- `src/relayspec/relay.py`
- `scripts/train_relay.py`

**Work:**

1. Implement relative interface MSE with accumulation-dtype-safe denominator.
2. Preserve raw MSE and historical objective only as named ablations.
3. Add pre-/post-fusion and rank configuration.
4. Add identity-provider, source-provider, relay-provider, and cache-transition
   tests.
5. Implement the Amdahl estimator with uncertainty propagation.

**Verify:**

```bash
pytest -q tests/test_losses.py tests/test_exactness.py tests/test_cost_model.py
```

Pass condition: analytic loss decomposition agrees numerically; greedy outputs
match native target exactly on fixtures; synthetic cost measurements recover
known speedup and break-even values.

### Task 4: Pin data, checkpoints, and leakage boundaries

**Create:**

- `scripts/check_prompt_overlap.py`
- `configs/train_math_4096.json`
- `configs/eval_manifest_full_v4.json`
- `tests/test_prompt_overlap.py`

**Work:**

1. Rebuild the existing 4,096 MATH training pool with source ID/revision and
   prompt hashes.
2. Build a disjoint development manifest.
3. Verify zero operator-preserving normalized-hash overlap with every test
   manifest.
4. Record Qwen3, DFlash, EAGLE-3, tokenizer, and scorer revisions.

**Verify:**

```bash
python scripts/check_prompt_overlap.py \
  --fit-manifest configs/train_math_4096.json \
  --evaluation-manifest configs/eval_manifest_full_v4.json \
  --output-json reports/final/PROMPT_OVERLAP_AUDIT.json
pytest -q tests/test_prompt_overlap.py
```

Pass condition: zero forbidden overlap and all manifests reproduce their
declared hashes.

### Task 5: Run the fast live profile before training sweeps

**Create:**

- `configs/profile_dflash_qwen3_8b_4gpu.yaml`
- `configs/profile_eagle3_qwen3_8b_4gpu.yaml`
- `slurm/profile_relayspec.sbatch`
- `reports/pretrain-cost-gate/`

**Work:** Run 32 fixed development prompts for native AR, official target
proposer, and optimized source reuse. Time every component with synchronized
CUDA events after the declared excluded warmup. Measure source-trunk fraction and the maximum
translator budget allowed by the break-even inequality.

**Verify:**

```bash
sbatch slurm/profile_relayspec.sbatch \
  configs/profile_dflash_qwen3_8b_4gpu.yaml
sbatch slurm/profile_relayspec.sbatch \
  configs/profile_eagle3_qwen3_8b_4gpu.yaml
python scripts/aggregate_results.py reports/pretrain-cost-gate
```

Pass condition: component fractions sum to measured end-to-end time within
profiling error; Amdahl ideal speed and acceptance threshold have paired CIs.

### Task 6: Select the 8B design on disjoint development data

**Create:**

- `configs/train_relay_dflash_qwen3_8b.yaml`
- `configs/train_relay_eagle3_qwen3_8b.yaml`
- `configs/validate_relay_qwen3_8b.yaml`
- `reports/design-selection/`

**Work:**

1. Run the short learning-rate range test.
2. Compare loss/interface choices.
3. Preserve the five-tap released interface rather than redesigning the frozen
   proposer.
4. Retain full rank after profiling establishes that relay arithmetic is below
   0.8% of request time.
5. Confirm the inherited DFlash proposal-length choice.
6. Freeze one configuration per proposer family before looking at test results.

Each of four GPUs runs one independent candidate when memory permits; otherwise
the model is sharded and candidates run sequentially. Never compare a
single-GPU candidate with a four-GPU candidate as a speed result.

**Verify:**

```bash
pytest -q
python scripts/audit_research_constants.py \
  --protocol configs/relayspec_protocol.yaml \
  configs/protocol_active/*.yaml
python scripts/analyze_profile.py \
  --input reports/design-selection/eagle3/relayspec-eagle3-selected-val-25553/benchmark-rank*.jsonl \
  --reference-method source_reuse_eagle3 \
  --candidate-method relay_eagle3 \
  --output-json reports/design-selection/eagle3/selected-analysis.json \
  --output-md reports/design-selection/eagle3/selected-analysis.md
```

Repeat the explicit analysis command for every immutable candidate artifact.
Pass condition: conformance tests pass and the selected candidate clears the
measured Amdahl gate. All alternatives remain in the ablation table.

### Task 7: Fit frozen 8B and 14B translators

**Create:**

- `configs/train_relay_dflash_qwen3_14b.yaml`
- `configs/train_relay_eagle3_qwen3_14b.yaml`
- immutable checkpoint directories under `artifacts/relays/`

**Work:** Train only the selected translator architecture. Target, source, and
proposer weights remain frozen. Save optimizer/config/data/code hashes and
training curves. Use the single protocol seed as a reproducibility identifier,
not as a claim about optimizer variance; system uncertainty is estimated over
hundreds of paired requests rather than three expensive refits.

**Verify:** checkpoint reload gives identical development tensors; no frozen
parameter changes; declared trainable parameter count matches state dict.

### Task 8: Run the single-request final suite

**Create:**

- family/target benchmark configs under `configs/protocol_active/`
- immutable raw results under `reports/final/{dflash,eagle3}-{8b,14b}-*/`

**Order:**

1. 16-prompt exactness and timing smoke test;
2. full MATH-500 for all four methods;
3. HumanEval and EvalPlus MBPP;
4. MT-Bench;
5. GSM8K-128 confirmation.

Run matched methods on the same GPU rank/request allocation. Interleave method
order by prompt to reduce temporal confounding. If output agreement fails, stop
and diagnose before scoring; retain the mismatch audit and require unchanged
official task outcomes rather than silently relabeling BF16 kernel divergence.

**Verify:** all prompts are present exactly once per method, the scorer
completes, source hashes match, target-only commit assertions hold, every
finite-precision sequence mismatch is retained and official quality is scored
independently, and component time reconciles with end-to-end time.

### Task 9: Run isolated resource measurements

**Create:**

- source-only and relay-only memory configs for each family/scale;
- `reports/final/*_MEMORY.{json,md}`.

Run source and relay in separate processes so incompatible model residency does
not contaminate allocated/reserved/peak memory. Record GPU telemetry, but do
not make an energy claim from coarse unsynchronized power polling. Production
concurrency remains explicitly outside the batch-one claim.

### Task 10: Produce paper tables from raw artifacts only

**Modify:**

- `reports/FINAL_RESULTS.md`
- `reports/EXECUTION_STATUS.md`
- `reports/amdahl-analysis.md`
- `docs/METHOD.md`
- `docs/RelaySpec_End_to_End_Research_Report.tex`
- `output/pdf/RelaySpec_End_to_End_Research_Report.pdf`

**Required main figures/tables:**

1. two-family/two-scale end-to-end speed, acceptance, exact agreement, quality;
2. native AR vs native proposer vs source reuse vs RelaySpec;
3. measured component shares and predicted-versus-observed Amdahl speedup;
4. acceptance survival by draft position;
5. adapter parameters, training GPU-hours, and memory savings;
6. minimal causal ablation table;
7. coefficient-free automatic profile-policy decisions across every breadth
   cell;
8. claim-evidence map with source status.

All generated table cells link to raw artifact IDs. The final DFlash tables use
the unified frozen-checkpoint reruns; earlier Part-1 numbers remain only as
historical ablation evidence.

---

## 8. Four-GPU schedule and expected wall time

| Stage | Four-GPU wall time estimate | Decision produced |
|---|---:|---|
| evidence audit and code contract | 4-7 h | constants and baselines frozen |
| EAGLE integration and equivalence tests | 4-8 h | second proposer works correctly |
| live profiles | 1-2 h | maximum possible gain and break-even acceptance |
| 8B sequential design selection | 4-7 h | one frozen variant per family |
| final 8B/14B translator fits | 3-6 h | immutable checkpoints |
| full single-request suites | 12-24 h | main paper results |
| isolated memory and final aggregation | 1-3 h | resource and paper artifacts |

Expected complete wall time is roughly **29-57 hours on exactly four GPUs**, not
including queue delay or initial checkpoint transfer. The first decisive answer
arrives after the 1-2 hour live profile: it shows how much total latency is
actually removable for EAGLE and whether any translator can plausibly win.

No expensive full benchmark is submitted before exactness, component
reconciliation, and the Amdahl gate pass.

---

## 9. Paper-level success criteria

The study supports a strong positive paper only if all of the following hold:

1. target-only verification is established by executable invariants and formal
   assumptions; finite-precision mismatch rate and official quality delta are
   fully reported rather than forced to zero;
2. RelaySpec improves source-reuse end-to-end throughput with a paired 95% CI
   above 1 for both DFlash and EAGLE-3 on the primary MATH-500 result;
3. the improvement appears at 8B and 14B, not only one cherry-picked scale;
4. measured gains are explained by the pre-registered Amdahl model;
5. source reuse and relay have unchanged official task quality, while native
   one-token AR differences caused by finite-precision kernel shapes are
   reported separately;
6. the translator is materially cheaper to train/store than a target-specific
   proposer, with measured numbers;
7. the novelty table distinguishes RelaySpec from DFlare, KV reuse, proposer
   steering, generic drafters, and target-specific EAGLE/DFlash training.

The headline is not a heuristic loss and not “KV sharing.” It is:

> **A frozen proposer can be transplanted across verifier scales by learning its
> exact consumed conditioning interface, eliminating a repeated source trunk;
> the resulting gain is predictable from removable runtime share and preserved
> acceptance, while target-only verification keeps decoding exact.**
