# Plan: making RelaySpec's relay adapt to an evolving/fine-tuned target

## Why this plan exists

Phase A (the SFT drift dose-response study) found something stronger than
expected: a frozen relay's speculative benefit collapses almost immediately
under fine-tuning (already near the floor by 50 LoRA steps, no worse at
1,000 steps), and a plain low-rank linear correction (`delta_rank`, already
implemented, E3 Stage B's mechanism) does not recover it. That is a real,
negative result on its own. This plan asks the next question directly: is
there *any* cheap, frozen-proposer-preserving fix that recovers acceptance
after drift, and if one exists, does it turn RelaySpec from a cross-model
transfer trick into a genuine post-training/RL-serving systems contribution
(the comparison point the user named is FastGRPO-style continual drafter
retraining)? We answer this with the checkpoints and infrastructure that
already exist, not new fine-tuning runs.

**Hard constraint carried through every experiment below:** reuse the five
already-fine-tuned Qwen3-8B/UltraChat/LoRA checkpoints (steps 50/150/400/
1000/2500) from Phase A. Do not train new target models. Every ablation
here is a different way of *fitting the relay* against those same fixed
checkpoints, not a new drift study.

## First, fix a real bug this plan depends on

Before any new ablation, the existing closed-form-vs-gradient-descent
ablation (Table 8) was invalid: the closed-form solver minimized plain,
unweighted ridge regression, while gradient descent minimizes
`relative_interface_mse`, a per-position-energy-weighted loss. These are
different objectives, so "closed-form underperforms gradient descent" was
never a clean comparison. Fixed in `scripts/fit_relay_closed_form.py` by
weighting each row by `1/||c_t||` before accumulating the normal equations,
which is provably the same optimum as the weighted loss (proven by a
stationary-point gradient check in
`tests/test_closed_form_weighting.py`, which also confirms the *old*
unweighted solve was demonstrably not a stationary point of the real loss).
The closed-form ablation must be rerun under the fix before its number goes
back in the paper; the direction of the conclusion may change since the
whole comparison was previously confounded.

## Experiment 1: residual nonlinear correction, not an MLP replacement

**Question**: does nonlinearity help the relay at all, or did the earlier
MLP ablation (which replaces the linear map entirely and lands at 51.7%
retention, the worst configuration measured) only prove that *discarding*
the linear solution is bad, not that nonlinearity itself is useless?

**Mechanism, already implemented** (`src/relayspec/relay.py`,
`TargetFeatureRelay`, `delta_nonlinear=True`): keep the frozen linear map
`R` as the backbone and add `c = R z + U sigma(V z)`, where `V` projects to
a rank-`delta_rank` bottleneck (128, matching the existing `delta_rank`
convention) and `U` projects back up, with a GELU between them. `U` is
zero-initialized exactly as the existing linear `delta_rank` mechanism
already is, so the nonlinear path starts as an exact no-op and can only
ever learn a residual, never override the linear solution. This reuses
E3 Stage B's exact training path (`train_relay.py`'s `delta_rank` branch,
freeze `projection`, train only the delta), with one new boolean flag.

**Runs** (all against the base relay, fitted before any fine-tuning, on
each of the 5 drift checkpoints, 128 prompts, matching Phase A's own
protocol exactly): linear delta (already run, Phase A's negative result)
vs. nonlinear delta (`delta_nonlinear=True`) at each checkpoint. Report
accepted-length retention for both, side by side, at every step.

**Registered success/failure criterion**: if nonlinear delta recovers
retention to a level linear delta could not (target: meaningfully above
the ~20% floor linear delta is stuck at, ideally toward the 90%+ range
un-drifted checkpoints show), that is this plan's central positive result.
If it doesn't move materially past linear delta's floor, that is also
reported directly: it would mean the failure is not a capacity problem in
the correction (nonlinearity did not help either), reinforcing the
Phase A finding that this is behavioral drift in the target's own
token distribution, not an expressivity gap in how the tensor is built.

## Experiment 2: does the relay align models, or align tasks?

**Question**: fitting currently uses 4,096 MATH prompts. If the relay is
really recovering a stable geometric alignment between two models'
representations (the paper's own framing), fitting data domain should not
matter much. If it is secretly doing something math-task-specific, a
domain mismatch would show up as a measurable gap.

**Runs**: refit the base (pre-drift) relay four ways — 4,096 MATH prompts
(existing, no change), 4,096 generic instruction/chat prompts (UltraChat,
already cached on this cluster from the SFT work, zero new data
engineering needed), a 50/50 mixed set, and a diversity-selected subset
(k-means or farthest-point sampling over target hidden states `z_t^T`,
testing whether ~512 well-chosen prompts match 4,096 random ones — this
directly tests EDA's data-efficiency argument for RelaySpec's own fitting
step). Evaluate all four on the *same* held-out MATH-500/breadth suite
already used everywhere else in the paper, so cross-domain generalization
is measured on a fixed, unbiased yardstick.

**This experiment is independent of drift** — it uses the un-fine-tuned
base target, and is really testing the paper's own "aligns models, not
tasks" claim, which today rests only on the code/chat breadth results,
never on varying *fitting* data. Cheap to run (same fitting compute as the
existing 4,096-example fits) and directly strengthens or falsifies a claim
already made in the paper's abstract.

## Experiment 3: adaptive layer gating (fast path vs. slow path)

**Question**: can a much cheaper "fast" adaptation (a handful of scalar or
per-dimension gates on the five tapped layers) absorb some of the drift
before a full relay/delta refit is needed?

**Mechanism**: `c = R[alpha_1 h_1; alpha_2 h_9; ...; alpha_5 h_33]`, five
extra scalars (or five per-dimension gate vectors for a richer version),
trained the same way a `delta_rank` correction is (frozen base, small
number of new parameters). This is architecturally the cheapest possible
correction — cheaper than `delta_rank` itself — so the natural evaluation
is: at each drift checkpoint, does gating alone recover anything, and if
so, how does its cost/benefit compare to the full nonlinear delta from
Experiment 1? This gives RelaySpec's continual-adaptation story a genuine
two-tier structure (cheap gates for small/early drift, the fuller delta
for larger persistent drift) if gating alone helps at all; if it does not,
that boundary is reported directly, the same way every other negative
result in this paper already is.

## Experiment 4: relation between LoRA weights and the relay's correction

**Question**: is there a direct, checkable relationship between the LoRA
update actually applied to the target (`Delta W = B A` at each fine-tuning
step) and whatever correction (linear or nonlinear delta) recovers
acceptance, if one does?

**Concrete, checkable sub-questions**, all computable from artifacts that
already exist (the 5 LoRA checkpoints, the base and drifted relay
checkpoints, no new training needed): does the fitted delta's dominant
singular directions align with the LoRA update's own singular directions
at the tapped layers (a subspace-overlap number, the same kind of metric
`analyze_relay_manifold.py` already computes for other comparisons)? Does
delta magnitude grow monotonically with LoRA update magnitude across the
five checkpoints, given Phase A already found retention does *not* grow
monotonically (the "cliff" result)? A negative answer here (no clean
relationship) is itself informative: it would mean the correction needed
is not a simple function of the fine-tuning update's own geometry.

## Domain-expert per-task linear mappers (separate, smaller thread)

A different, task-side question from all of the above: instead of one
relay per target, fit one relay per (target, task) pair as a form of
task-conditioned "expert" mapping, and measure whether task-specific
fitting data raises acceptance retention on that task specifically,
compared to the single generic relay used everywhere else in this paper.
This reuses the exact same fitting code with a task-specific manifest
in place of the generic one; the only new work is selecting per-task
fitting prompts (GSM8K/HumanEval/MBPP/MT-Bench, already the paper's own
breadth-evaluation tasks) and comparing retention with vs. without a
task-specific fit. Lower priority than Experiments 1-4 above, since it
does not bear on the drift story directly, but cheap to run once the
drift-focused queue clears and directly extends the paper's existing
breadth results.

## How this would extend the paper

If Experiment 1 lands (nonlinear delta meaningfully recovers acceptance
after drift), the paper gains a second half with its own headline: not
just "one linear map survives a model change," but "a small, cheap,
frozen-proposer-preserving correction survives a target *changing over
time* under ordinary fine-tuning," with Experiment 3 giving a natural
fast/slow adaptation story and Experiment 4 giving a mechanistic account
of why. If Experiment 1 does not land, Phase A's existing negative result
still stands as reported (drift breaks frozen speculative decoding fast
and a plain correction cannot fix it), which is itself a valid, if less
constructive, ICLR-level finding, and Experiments 2-3 (data-domain
generality, gating) remain independently useful strengthening for Part 1's
existing claims regardless of how Experiment 1 turns out. Either outcome
is reported plainly; nothing here is contingent on getting a positive
result to be worth writing up.

We do not have RL-training infrastructure in this repo and are not
building any as part of this plan (Phase B, the GRPO proof of concept, was
explicitly deferred pending Phase A and remains deferred). The comparison
to FastGRPO-style continual drafter retraining stays qualitative
(different mechanism: those systems retrain the drafter itself; this one
never touches the drafter, only a small linear/nonlinear map on top of it)
until real RL infrastructure exists to measure it directly; we will not
fabricate a quantitative comparison to a system we have not run.

## Execution order and resourcing

Single 4-GPU lane throughout, per standing instruction. Order: (1) rerun
the fixed closed-form ablation (already queued), (2) Experiment 1 across
all 5 checkpoints (highest priority, directly answers the plan's central
question), (3) Experiment 4's analysis (no training needed, pure analysis
of existing checkpoints, can run any time GPUs are busy with something
else since it is comparatively cheap), (4) Experiment 2 (independent of
drift, can run in parallel conceptually but still single-queued
mechanically), (5) Experiment 3, (6) the domain-expert thread, lowest
priority.
