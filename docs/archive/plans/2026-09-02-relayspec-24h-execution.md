# RelaySpec: 24-hour execution plan

Date: 2026-09-02. Companion to the source-free plan and the implementation
plan. This is the schedule, with hour offsets from start.

Operating assumptions, all verified:

- 8 GPUs on `turing`, jobs stay at 4 GPUs each, so two concurrent job slots.
- Qwen3-4B, 8B, 14B and both DFlash checkpoints are already cached from the
  published runs. **No new model downloads.** This is the single biggest
  reason the 24-hour window sticks to Qwen3 and DFlash.
- Local suite is 134 tests in about 20 seconds on CPU, so code correctness is
  established locally, not on the cluster.

Ground rule the user set: no smoke tests for their own sake. The 32-prompt
development cells here are not smoke tests. They are the registered gate from
the main plan and they produce reportable numbers. Every GPU job below yields
a result that goes in the paper or kills a variant.

---

## What will NOT happen in 24 hours, decided up front

Stated so nobody waits for it:

- **EAGLE-3 anything.** Task 4 (differentiable conditioned forward through a
  cache-mutating chain) is the hard code task in the whole plan. Attempting it
  under time pressure risks silently wrong gradients. DFlash only.
- **On-policy S1.** Needs Task 5 plus 1.5 to 2 GPU-hours per fit.
- **X1 and X2 cross-family.** Blocked on a checkpoint survey and then on
  downloading a new family's weights, which alone can eat hours.
- **E3 descendant transfer.** Blocked on the same checkpoint decision.
- **E6 native-AR sweep and E9 vLLM.** Deferred by design.

---

## Hour 0 to 1: launch GPU work that has no code dependency

Two jobs go out immediately, in both slots, before any new code exists.

**Slot 1, E7 DFlash isolated memory.** Write two configs modelled exactly on
`configs/protocol_active/eagle3_qwen3_8b_memory_{source,relay}_4gpu.yaml`,
switched to `proposer.family: dflash` with one method per file. Four jobs
total across both scales, about 6 minutes each. Closes a confirmed gap: the
memory column is currently EAGLE-only because this was planned in the original
runbook and never executed.

Memory measurement is per-card peak allocation, so it is unaffected by an
unrelated job on the other four cards. Safe to run concurrently.

**Slot 2, E4-P0 representation analysis.** Write the analysis script and run
it. Load existing `W_8B` and `W_14B`, run both on identical text, and report
cosine similarity, CKA and principal-subspace overlap of the resulting
`c-hat` trajectories in the shared 2,560-dimensional interface. No training.
This is the free half of the highest-upside experiment and it informs whether
E4-P1 is worth its slot later today.

**Deliverable by hour 1:** the DFlash memory numbers and a first read on
whether the two relays are converging on a shared representation.

---

## Hour 0 to 4: critical-path code, CPU only, parallel with the above

Order matters. The goal is one working objective end to end, launched, before
the remaining objectives are wired.

1. **Task 0**, 10 minutes. Fix the `expected_accepted_length_surrogate`
   docstring. It currently asserts an identity that only holds under sampling,
   not under the deployed greedy rule.
2. **Task 1**, 30 minutes. Hard-accept diagnostic in
   `src/relayspec/metrics.py`, wired into the per-step metrics record. Every L3
   run needs `A_hard` logged next to `A_soft`, because the surrogate is
   licensed empirically.
3. **Task 3, L1 only first**, 90 minutes. Add `training.supervision` with
   `source_interface` as the untouched default and `target_verifier` as the
   new path. Wire **only** `greedy_agreement_ce` to begin with, because it is
   the simplest correct objective and it unblocks the pipeline. Verify locally
   that `source_interface` produces bit-identical losses and that
   `target_verifier` never calls the source trunk (stub that raises).
4. **Launch E1-L1 the moment step 3 is green.** Do not wait for L2 and L3.
5. **Task 3 remainder**, 60 minutes, while L1 runs. Wire `proposal_kl_to_target`
   (L2), `accepted_prefix_surrogate` (L3), plus
   `training.warm_start_checkpoint` (mitigation 1) and
   `training.regression_anchor_weight` (mitigation 2) for L4.
6. **E1 configs**, 30 minutes. Five fit configs plus five 32-prompt
   development benchmark configs, all DFlash-8B, in
   `configs/protocol_next/`. Keep `fit_examples == steps * world_size`, so
   4,096 equals 1,024 steps across 4 ranks.

---

## Hour 4 to 7: E1 development ladder, DFlash-8B

Five variants: L0 (source-supervised control, reproduces the published
number), L1, L2, L3, L4.

Per variant: one fit at about 2 minutes, one 32-prompt development cell at
about 7 minutes. Nine minutes serial, so five variants across two slots is
roughly 25 to 30 minutes of pure compute. Budget 3 hours for launch overhead,
failures and reruns.

**Gate applied here, from the main plan Section 7.** A source-free variant
passes if its acceptance retention is within about 2 points of L0's. Record
`A_hard` and `A_soft` for every variant, plus the `c-hat` drift metric, so the
optimization-surface risk is visible rather than inferred.

**Decision at hour 7.** Either at least one of L1, L2, L3 clears the gate and
the source-free framing is live, or it does not and L4 becomes the headline
with the framing narrowed to "source-trunk-free fitting". Write the decision
down before proceeding.

---

## Hour 6 to 8: Task 2 relay variants, CPU, overlapping the above

Add `rank`, `shared_decoder` and `delta` modes to
`src/relayspec/relay.py`, with `parameter_summary()`. Default construction
must stay numerically identical. This unblocks both E4-P1 and E5 for the
afternoon.

---

## Hour 7 to 10: full MATH-500 for survivors, DFlash-8B

Promote every gate-clearing variant, at most three, to the full 500-prompt
paired cell. DFlash-8B full is 18.3 minutes of measured generation plus about
5 minutes of loading, so roughly 25 minutes per variant, two at a time.

**This produces the first real headline number of the iteration:** a
source-free relay's end-to-end speedup against optimized source reuse on
MATH-500, directly comparable to the published 1.483x because the protocol is
unchanged.

---

## Hour 10 to 14: E1 ladder on DFlash-14B

Same five variants, or just the gate winners plus L0 if time is tight. 14B is
where the two negative code cells live, so the source-free objective's
behaviour at 14B is more informative than at 8B.

Development cells first, then full MATH for the winner. DFlash-14B full is
26.4 minutes plus loading, so about 32 minutes per variant.

---

## Hour 10 to 13: E4-P1, overlapping, decides Branch A or B

The highest-upside experiment in the plan. Train a shared proposer-facing
decoder using one target, then feed it `u` derived from the other target and
measure zero-shot acceptance loss.

Two fits at 2 minutes each, plus cross-target development cells at 7 minutes
each. Under an hour of compute, so it fits alongside the 14B ladder.

**Threshold, fixed in the config before the first fit, not after seeing the
number:** Branch A if zero-shot transfer stays within about 3 points of
acceptance retention. Branch A means the canonical interface becomes the
central contribution and the paper is reorganized around it. Branch B means
this becomes an appendix negative result.

---

## Hour 14 to 20: E5 capacity sweep and E2 first pass

**Slot 1, E5.** Rank sweep on the winning objective, plus a tap-count and
tap-selection sweep. About eight fits at 2 minutes and eight development
cells at 7 minutes, so roughly 75 minutes of compute across the slot. Answers
the "why 52 to 66M parameters" question directly and feeds the low-rank
constraint that E3's delta bridges will need later.

**Slot 2, E2 first pass, 14B code only.** Build a math-plus-code fitting
manifest from training splits only, re-run both prompt audits, fit, then
evaluate on 14B HumanEval and MBPP. This is the direct attack on the paper's
two negative cells. Scoping to 14B code is the difference between roughly 4
hours and roughly 14.

---

## Hour 20 to 24: aggregate and write

No new GPU launches after hour 20, so anything still running has time to
finish.

1. Run the aggregation and matrix builders over the new result directories.
2. Extend `scripts/build_iclr_paper_assets.py` for the objective-ladder table,
   the per-variant fit-cost table, and the `A_hard` versus `A_soft`
   correlation plot that licenses L3.
3. Record the two decisions of the day in
   `docs/research/relayspec-decision-register.md`: the supervision gate
   outcome and the Branch A or B outcome.
4. Write the honest status: which variants cleared, which did not, and what
   the next 24 hours should target.

---

## Expected GPU consumption

| Block | GPU-hours |
|---|---|
| E7 DFlash memory, 4 jobs | 1.6 |
| E4-P0 analysis | 0.5 |
| E1 development ladder, 8B, 5 variants | 3.2 |
| E1 full MATH, up to 3 survivors at 8B | 4.8 |
| E1 ladder plus full MATH, 14B | 6.0 |
| E4-P1 | 1.5 |
| E5 sweep | 5.0 |
| E2 first pass, 14B code | 4.0 |
| **Total** | **about 27** |

Against 8 GPUs for 24 hours the ceiling is 192 GPU-hours, so this plan uses
roughly 14 percent of nominal capacity. That is deliberate. The binding
constraint today is the 4 hours of critical-path code and the serialization of
gate decisions, not GPU availability. If the code lands early, pull E1's
EAGLE-3 ladder forward only if Task 4 is genuinely finished and tested, and
otherwise widen E2 to both scales instead.

---

## Risks specific to this window

1. **Task 3 overruns.** It is the critical path. Mitigation is already in the
   schedule: ship L1 alone, launch it, then wire L2, L3 and L4 while it runs.
2. **A source-free variant diverges.** Expected failure mode from the main
   plan Section 3.2. Warm-start from the published L0 checkpoint is already a
   config key, so the fallback is one field, not a code change.
3. **Concurrency perturbs timing.** All throughput-critical cells here are
   full MATH cells. If two concurrent 4-GPU jobs are suspect, serialize the
   full-MATH promotions and keep concurrency for fits, development cells and
   memory jobs, which are not throughput-critical. Cost of that fallback is
   about 2 extra hours, absorbed by the hour 7 to 14 window.
4. **First-run config errors.** The single realistic reason to lose an hour.
   `tests/test_configs.py` schema-validates every new config locally, so run
   it before every launch batch.
