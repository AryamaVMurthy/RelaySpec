# RelaySpec source-free retargeting: implementation plan

Date: 2026-09-02. Companion to
`2026-09-02-relayspec-source-free-adaptive-relay-plan.md`, which holds the
what and the why. This document holds the how, in execution order, with
interfaces and acceptance tests.

Ground rules for every task below:

- The default code path must stay byte-identical in behaviour, so every
  published number in `paper/iclr2027/` stays regenerable.
  `configs/protocol_active/` is never edited. New configs go in
  `configs/protocol_next/`.
- Every task has a local acceptance test that runs on CPU. The suite is 127
  tests in about 14 seconds, so nothing reaches the cluster untested.
- GPU work runs on `turing` only, partition `u22`, `node01`,
  `--gres=gpu:4`. Never the local laptop GPU.
- `scripts/train_relay.py:71-72` hard-asserts `WORLD_SIZE == 4` and
  `:176-181` requires `fit_examples == steps * world_size`. Preserve that
  arithmetic in every new config.

---

## Task 0. Correct the L3 docstring (blocking, 10 minutes)

**Status of prior work.** `src/relayspec/losses.py` already contains
`_check_verifier_shapes`, `greedy_agreement_ce` and
`expected_accepted_length_surrogate`, with 8 tests in
`tests/test_losses.py` (15 passing in that file). The additions are purely
additive and change no existing call site.

**Defect to fix.** The docstring of `expected_accepted_length_surrogate`
currently asserts that the greedy accepted prefix length *is*
`sum_i prod_{j<=i} a_j`. Per plan Section 3.1.1 that identity holds only under
a sampled-proposal regime, not under the deployed greedy-argmax rule. The
docstring overclaims and must be rewritten to state: the exact identity under
sampling, the relaxation status under greedy, the absence of a general
inequality in either direction, and the `a_j > 1/2` sufficiency condition.

**Also rename** for honesty: `expected_accepted_length_surrogate` stays, but
add module-level documentation distinguishing `A_soft` from `A_hard` using the
plan's notation.

**Acceptance test.** Add a test asserting the surrogate and a hard-accept
count disagree on a constructed near-tie case, which pins the documented
non-equivalence in code rather than only in prose.

---

## Task 1. Hard-accept diagnostic (prerequisite for every L3 run)

Section 3.1.1 licenses L3 empirically, so `A_hard` must be measurable
wherever `A_soft` is optimized.

**Add** to `src/relayspec/metrics.py`: a function taking proposal logits and
verifier tokens and returning the hard accepted-prefix length, the per-position
hard acceptance indicators, and the mean agreement mass. No gradients.

**Wire** it into the training metrics record written per step at
`scripts/train_relay.py:326` onward, so every fit logs `A_hard` and `A_soft`
side by side.

**Acceptance test.** Hard length equals the surrogate exactly when all
agreement masses are one, is zero when the first position disagrees regardless
of later positions, and is insensitive to changes in non-argmax logits.

---

## Task 2. Relay module variants

**File** `src/relayspec/relay.py` (currently 57 lines, one `nn.Linear`).

Add three constructor options, all defaulting to current behaviour:

- `rank: int | None = None`. When set, factor the projection as two matrices
  through a rank-`r` bottleneck. Needed by E5 and by E4-P2.
- `shared_decoder: nn.Module | None = None`. When provided, the module owns
  only the per-target encoder and calls the supplied shared decoder. Needed by
  E4-P1 and E4-P2.
- `base_projection: Tensor | None = None` plus `delta_rank: int`. When set,
  the base weight is registered as a frozen buffer and only a low-rank delta
  is trainable. Needed by E3 Stage B.

**Constraint.** Parameter count must remain reportable per variant, because
the paper quotes 52.4M and 65.5M and E5 exists to interrogate those numbers.
Expose a `parameter_summary()` returning trainable and frozen counts.

**Acceptance tests.** Default construction is numerically identical to the
current module. Rank-`r` construction has the expected parameter count. Delta
mode leaves the base weight unchanged after an optimizer step, and reduces to
the base map when the delta is zero-initialized. Shared-decoder mode routes
gradients to the encoder only when the decoder is frozen.

---

## Task 3. Verifier supervision in the training loop

**File** `scripts/train_relay.py`. This is the largest change and the one that
must not disturb the default path.

**Add config key** `training.supervision`, values `source_interface` (default,
current behaviour exactly) and `target_verifier`.

Under `target_verifier`:

1. Delete the source trunk forward at `:214-218` from the executed path, and
   do not construct `teacher_context` at `:224-228`. The source model is still
   loaded only if the proposer needs its embedding or head, which Task 8
   resolves.
2. Stop passing `logits_to_keep=1` on the target call at `:229-238` so
   per-position logits are available. Derive verifier tokens by argmax over
   the block positions, matching `greedy_sample` in
   `src/relayspec/generation.py:19-24` so training and benchmarking agree on
   the accept rule.
3. Build the proposal block exactly as the dormant KL path does at
   `:242-254`, then run the conditioned proposer forward and take logits.
4. Apply the Axis A objective selected by `training.feature_objective`,
   extended to accept the new values `greedy_agreement_ce`,
   `proposal_kl_to_target` and `accepted_prefix_surrogate`.
5. Log `A_hard`, `A_soft`, and the `c-hat` norm and direction drift required by
   plan Section 3.2 mitigation 3.

**Template to follow.** The existing `proposal_kl_weight` path at `:181-193`,
`:242-265` and `:281-300` already runs the frozen draft transformer on a
relay-produced context and takes logits through
`_conditioned_dflash_forward` (`src/relayspec/generation.py:246-268`). The
change is what the teacher is, not whether the plumbing exists.

**Add config keys** for the mitigations: `training.warm_start_checkpoint`
(path to an L0 relay, implements mitigation 1) and
`training.regression_anchor_weight` (default 0.0, implements mitigation 2).

**Acceptance tests.** With `supervision: source_interface` the step function
produces identical loss values to the current implementation on a fixed seed
and a stub model. With `supervision: target_verifier` no source trunk forward
is invoked, verified by a stub that raises if called. Warm start loads a
checkpoint and reproduces its outputs at step zero. Anchor weight zero gives
exactly the unanchored loss.

---

## Task 4. EAGLE-3 differentiable conditioned forward

**File** `src/relayspec/eagle3.py`.

Today `scripts/train_relay.py:190-191` raises for any non-DFlash proposal
training, so without this task the entire headline result is DFlash-only,
which weakens the cross-family argument that the current paper's abstract
already makes.

**Add** a teacher-forced conditioned forward built from
`Eagle3Backend.initialize` and `propose` (`:145-229`). It must accept a
relay-produced context with gradients attached, run the length-7 chain with
teacher-forced tokens rather than sampled ones, and return per-position logits
via `draft.compute_logits`. The chain's cache handling is the hard part,
because `propose` is written for inference and mutates cache state.

**Risk and fallback.** If cache mutation makes gradients unreliable, fall back
to a single-step conditioned forward, supervise position one only, and record
the reduction in scope explicitly. Plan Section 8 already registers this
outcome.

**Acceptance tests.** Gradients reach the relay input and are zero for
proposer parameters. Teacher-forced logits at position one match the
inference path's logits for the same context. Chain length is configurable and
matches the benchmark default of 7.

---

## Task 5. On-policy state collection harness (Axis B, S1)

**New file** `src/relayspec/onpolicy.py`.

Provide a collector that, given a prompt, runs the real decode loop with the
current relay, records the visited committed prefixes and the target's
verifier tokens at each proposal block, and returns them as training states.
The rollout is fully detached. Gradients are taken only later, when the
training step recomputes the proposer forward at a collected state.

**Reuse** the generation path in `src/relayspec/generation.py` for collection
rather than writing a second decoder, so the collected state distribution is
by construction the deployed one.

**Cost control.** Expose `collect_every` and `states_per_rollout` so the
Phase 2 sweep can trade fit cost against on-policy fidelity, and log
wall-clock fit time per variant, since plan Section 3.1 requires per-variant
cost reporting.

**Acceptance tests.** Collected prefixes are a prefix chain. No parameter
receives a gradient during collection. Deterministic under a fixed seed with
greedy decoding. Collector and benchmark harness agree on accepted-token
counts for the same prompt and relay.

---

## Task 6. Configs

**New directory** `configs/protocol_next/`, mirroring the naming conventions
of `protocol_active`.

Required sets:

- E1: five objective variants for DFlash-8B under S0, plus the matching
  development-gate benchmark configs.
- E1b: the winning objective under S1.
- E2: three fitting-data compositions, with new manifests.
- E3: descendant zero-shot benchmark config and Stage B delta fit config.
- E3b: the architecturally shifted target, pending the Section 3.4.1 survey.
- E4: P1 shared-decoder fit configs for two targets, plus the cross-target
  zero-shot benchmark config.
- E5: rank sweep and tap-selection sweep fit configs.
- E7: `dflash_qwen3_{8b,14b}_memory_{source,relay}_4gpu.yaml`, modelled
  exactly on the existing EAGLE-3 memory pair, one method per file so the
  processes stay isolated.

**Acceptance test.** Extend `tests/test_configs.py` so every new config is
schema-validated, the `fit_examples == steps * world_size` invariant is
checked, and `gpu_count` is 4 everywhere.

---

## Task 7. Fitting manifests and audits for E2

**New manifests** for math plus code, and math plus code plus chat, drawn
from training splits only, sized to keep the 4-rank arithmetic exact.

**Re-run** `scripts/check_prompt_overlap.py` and
`scripts/audit_prompt_similarity.py` for every new manifest and record the
overlap and similarity numbers per manifest, as plan Section 3.3 requires.

**Acceptance test.** Zero exact overlap against all 1,250 evaluation records
for every new manifest, asserted in `tests/test_prompt_overlap.py` style.

---

## Task 8. Provenance verification (gates the headline sentence)

Determine whether the released DFlash artifact ships its own token embedding
and LM head, or genuinely borrows Qwen3-4B's, given that
`scripts/train_relay.py:250` uses `source.model.embed_tokens` and `:263`,
`:293` use `source.lm_head`.

**Deliverable.** An entry in `docs/research/relayspec-decision-register.md`
recording the finding with the checked file locators, and a decision on which
of the two headline phrasings in plan Section 1.1 is permitted.

This is prose work, not code, but it blocks the abstract, so it happens in
Phase 0.

---

## Task 9. Verification tasks that gate claims

Three Phase 0 items with no code:

1. Read SD-squared's frozen-drafter ablation closely and confirm or refute the
   three differentiators in plan Section 3.6. If they do not hold, positioning
   changes before submission.
2. Source Draft-OPD, OnlineSPEC and Learning to Draft through the source-log
   process, recording status, venue, primary URL and checked claims. Add
   `references.bib` entries only after verification. Do not write entries from
   memory.
3. Survey checkpoints for a tokenizer-compatible, architecturally shifted,
   single-card Qwen3 derivative per plan Section 3.4.1, and record the finding
   either way.

---

## Task 11. Cross-family token bridging (new, gates X1)

**New file** `src/relayspec/vocab_bridge.py`, plus a benchmark path change.

Implements S-A from plan Section 3.5b, string-level verification, which needs
no new learned parameters.

Required pieces:

1. `propose_as_text(proposal_token_ids, source_tokenizer) -> str`. Decode a
   proposed block from the proposer's vocabulary to text. Must handle partial
   or dangling byte-level BPE fragments at the block boundary rather than
   producing replacement characters.
2. `longest_agreeing_prefix(proposal_text, target_greedy_text) -> int`.
   Character-granular longest common prefix, returned as a count of characters
   agreed.
3. `commit_in_target_tokens(agreed_text, target_tokenizer) -> list[int]`.
   Re-encode the agreed span in the target vocabulary, and truncate to the last
   whole target token so the committed prefix never ends mid-token. The
   remainder is dropped, which is where the acceptance cost of S-A comes from,
   and it must be logged so the cost is measurable rather than hidden.
4. Benchmark integration in `src/relayspec/generation.py`: when target and
   proposer tokenizers differ, route acceptance through the three functions
   above instead of the current token-equality path.

**Optional refinement, S-B.** `intersection_map(source_tokenizer,
target_tokenizer)` returning an injective partial map over tokens whose
decoded strings are identical, with coverage statistics. Where a proposed
token is in the intersection, keep token-granular acceptance. Fall back to
S-A only for the remainder. Report intersection coverage as a table, since it
predicts how much acceptance S-B recovers over S-A.

**Correctness note for the paper.** Proposition 1's induction carries over at
string granularity, because the target still produces every committed
character. This is a restatement, not a new proof, and it belongs next to the
existing proposition rather than in an appendix.

**Acceptance tests, all CPU.** Round-trip identity when both tokenizers are
the same, so the same-family path is provably unchanged. Correct handling of a
proposal that agrees on a prefix ending mid-target-token. No replacement
characters for any block boundary in a fixed corpus. Committed text is always
a prefix of the target's greedy text. Intersection map is injective and its
reported coverage matches a brute-force recount.

---

## Task 10. Analysis and paper assets

- E4-P0 analysis script comparing `c-hat` trajectories from the existing
  `W_8B` and `W_14B` in the shared 2,560-dimensional interface, reporting
  cosine similarity, CKA and principal-subspace overlap. No training, no new
  GPU jobs beyond two target forwards.
- Extend `scripts/build_iclr_paper_assets.py` for the new tables: the
  restructured Table 6 from plan Section 5.3, the objective-ladder table, the
  per-variant fit-cost table, and the `A_hard` versus `A_soft` correlation
  plot that licenses L3.
- Citation insertions and the Table 6 rebuild in
  `paper/iclr2027/relayspec_iclr2027.tex`, subject to the nine-page
  limit. Note that Table 6 gains two columns, so budget the space before
  writing.

**Acceptance test.** `tests/test_iclr_paper_assets.py` extended for each new
generated asset, and `tests/test_iclr_manuscript_audit.py` still passes,
including the nine-page check.

---

## GPU budget

### Measured unit costs, derived from existing artifacts

Everything below is derived from recorded runs, not guessed.

| Unit | Basis | Wall-clock on 4 GPUs | GPU-hours |
|---|---|---|---|
| One relay fit, S0, 1024 steps | 85.6 s recorded in `reports/training/dflash-relative-8b/relay-training-summary.json`, 86 to 119 s across configs | about 2 min | **0.13** |
| Model loading per job | 8B or 14B target, plus 4B source, plus proposer, warm cache | about 5 min | **0.33** |
| Full paired MATH-500 cell, DFlash-8B | 390,145 tokens at 148.3 and 220.0 tok/s | 18.3 min plus loading | **1.6** |
| Full paired MATH-500 cell, DFlash-14B | 387,725 tokens at 110.4 and 137.6 tok/s | 26.4 min plus loading | **2.1** |
| Full paired MATH-500 cell, EAGLE-8B | 387,600 tokens at 103.4 and 82.8 tok/s | 35.1 min plus loading | **2.7** |
| Full paired MATH-500 cell, EAGLE-14B | 383,071 tokens at 70.3 and 65.0 tok/s | 47.3 min plus loading | **3.5** |
| Development gate cell, 32 prompts | 6.4 percent of a full cell, loading dominates | about 7 min | **0.5** |
| One 4-task breadth sweep | shorter generations than MATH, from recorded p50 latencies | 15 to 25 min plus loading | **1.5 to 2.0** |
| One isolated memory job, 8 prompts | matches the EAGLE-3 memory pair | about 6 min | **0.4** |
| One on-policy fit, S1 | rollout generation during fitting, roughly 10 to 15x an S0 fit | 20 to 30 min | **1.5 to 2.0** |

### How to use 8 GPUs without invalidating published numbers

This matters more than the raw budget. Three constraints:

1. **Timing-critical jobs must stay at 4 GPUs.** Every published cell is
   "exactly four RTX 6000 Ada GPUs", and the measured quantity is per-request
   latency at batch one with one model copy per rank. Running a benchmark at 8
   ranks does not change the metric in principle, but 8 concurrent BF16
   processes on one node can hit power or thermal limits, and any clock
   throttling changes per-request latency directly. That would silently
   corrupt the comparison against all 20 existing cells.
2. **Relay fits must not move to 8 GPUs.** `scripts/train_relay.py:71-72`
   asserts `WORLD_SIZE == 4` and `:176-181` requires
   `fit_examples == steps * world_size`. At 8 ranks, 4,096 examples becomes
   512 steps of effective batch 8 instead of 1,024 steps of effective batch 4.
   That changes the trained artifact, not only the speed, so 8-GPU fits are
   not comparable to any published fit. Fits cost 0.13 GPU-hours, so there is
   nothing to gain by breaking this.
3. **The gain from 8 GPUs is job-level parallelism, not per-job speedup.**
   Run two independent 4-GPU jobs at once and the campaign finishes in about
   half the wall-clock, with every job still measured under the published
   4-GPU protocol.

**Blocking validation, Task -1, about 1 GPU-hour.** Before relying on
concurrency, re-run one already-published development cell alone on 4 GPUs,
then re-run it again while a second unrelated 4-GPU job occupies the other
four cards. If the tokens-per-second figure lands inside the existing
bootstrap interval, two-at-a-time is safe for the whole campaign. If it does
not, every timing job serializes and the wall-clock estimates below double.
This single cheap check de-risks the entire schedule, so it runs first.

### Per-phase budget

| Phase | Contents | GPU-hours |
|---|---|---|
| Task -1 | Concurrency perturbation check | **1** |
| Phase 0 | All code and verification tasks are CPU-only. E4-P0 needs two target forwards | **0.5** |
| Phase 1 | E1 dev ladder, 5 variants on DFlash-8B (5 x 0.63); full MATH for about 2 survivors (2 x 1.6); E4-P1 dev (1.5, or 4 if confirmed on full MATH); E7 DFlash memory, 4 jobs (1.6) | **10 to 13** |
| Phase 2 | E1 completion, EAGLE-3 ladder and 14B ladder dev plus full MATH for winners (about 14); E1b on-policy, 2 to 4 fits plus dev (6 to 10); E5 rank and tap sweep, about 8 fits plus dev (5); E3b if a target exists (4) | **29 to 37** |
| Phase 3 | E2 task composition, 14B only (about 14) or both scales and families (about 40); E3 descendant, Stage A plus Stage B (5.5), plus 1 to 3 if the LoRA descendant is self-made | **20 to 48** |
| Phase 4 | E6 native AR across 16 breadth cells, AR is slower per token so this is not cheap (about 10); audits are CPU-only | **10** |
| X2 | Within-family replication in a non-Qwen family: 1 to 2 fits (0.3), MATH cell (about 2.5), one code task (about 1), plus first-time model caching | **5 to 7** |
| X1 | Cross-family retargeting: Task 11 is CPU work, then 1 fit (0.13), MATH cell (2.5), one code task (1), plus an S-A versus S-B acceptance comparison (1) | **5 to 6** |
| **Total, excluding E9** | | **80 to 123** |
| E9 vLLM or SGLang | GPU cost is modest once it works. The cost is engineering, not compute | **5 to 10 plus engineering** |

### Wall-clock translation

At 8 GPUs running two 4-GPU jobs concurrently, 70 to 110 GPU-hours is 9 to 14
hours of perfectly packed cluster time. Realistically, with queue waits,
failed runs, reruns after config errors, and jobs that cannot overlap, apply a
2x to 3x overhead factor:

- **Optimistic, Phase 0 to Phase 2 only:** about 40 GPU-hours, so roughly
  1 to 1.5 days of cluster time.
- **Full campaign through Phase 4:** 70 to 110 GPU-hours, so roughly
  2 to 4 days of cluster time.
- **If the Task -1 concurrency check fails:** double the wall-clock, so
  4 to 8 days, because every timing job serializes onto 4 of the 8 cards.

### Cost control levers, in the order to pull them

1. Gate hard at the development stage. A dev cell is 0.5 GPU-hours against
   1.6 to 3.5 for a full cell, so never promote a variant to full MATH before
   it clears the dev gate.
2. Scope E2 to 14B first. That is where the negative cells are, and it is the
   difference between 14 and 40 GPU-hours.
3. Treat E1b on-policy as optional. It is 6 to 10 GPU-hours and, per plan
   Section 3.1, it also threatens the headline fit-cost claim. If S0 already
   clears the Phase 1 gate, S1 is a nice-to-have.
4. Defer E6 to last. It is 10 GPU-hours of consistency cleanup that changes no
   scientific claim.
5. E9 stays out of the budget until it is a committed decision.

---

## Execution order

**Task -1, first, about 1 GPU-hour.** The concurrency perturbation check from
the GPU budget section. Its result decides whether the rest of the campaign
runs two-at-a-time or serialized, so it gates every schedule estimate below.

**Phase 0, no GPU except 0.5 GPU-hours for E4-P0.** Task 0, Task 1, Task 2,
Task 3, Task 4, Task 6 partial (E1 and E7 configs), Task 8, Task 9, Task 10
first bullet. Everything else here is locally testable. Exit condition: full
suite green, and the three Task 9 findings recorded.

**Phase 1, 10 to 13 GPU-hours, two gates.** E1 under S0 on DFlash-8B,
development gate first. E4-P1. E7, which is four short jobs and closes a known
gap cheaply. Exit conditions: the supervision gate from plan Section 7, and
the Branch A versus Branch B decision from plan Section 3.5.

**Phase 2.** Task 5 and E1b. E1 completion for EAGLE-3 and 14B. E5. E3b if
Task 9 item 3 found a target.

**Phase 3.** E2 and E3, using the Phase 1 winning objective.

**Phase 4.** E6, the figure and equation audit, Task 10 remaining bullets,
then E9 only with committed engineering time.

---

## What is not in this plan

- Any change to `configs/protocol_active/` or to published report directories.
- Any tensor-parallel or model-sharding work, which plan Section 3.4.1 shows
  would invalidate timing comparability across every published cell.
- Any vLLM or SGLang integration, which stays a separate decision.
- Any new bibliography entry written from memory rather than verification.
