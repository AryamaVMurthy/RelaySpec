# Plan: make cross-family retargeting (X1) actually work

Date: 2026-09-03. Supersedes nothing; extends
`docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md` section
3.5b after that section's X1 was implemented and measured (decision
register, "X2 and X1: cross-family, executed per explicit user direction").

## 1. Where X1 actually stands, and why

Measured result: mean accepted tokens per cycle 1.185 (essentially the
forced floor of 1.0), 0.871x throughput versus plain autoregressive
decoding, and only 50% exact-text agreement with plain decoding on the same
prompts (every other method this project has measured is 100%, only
approximate near ties in a handful of EAGLE-3 rows, see
`app:correctness`). The pipeline runs without crashing. It is not useful.

Two independent problems are plausible, and they have different fixes:

**Problem A: the relay was fit almost blind.** Cross-family fitting only
had one usable supervision signal per training example: the last position,
because the source and target tokenize the same text into sequences of
different lengths with no known position-by-position correspondence
(`train_relay.py`, `cross_family` branch). Every same-family fit in this
project uses roughly 190 positions of supervision per example (the whole
sequence). Cross-family used 1. That is not a small handicap, it is
throwing away over 99% of the training signal, and it alone could explain
near-floor acceptance without any bug at all.

**Problem B: the discrete bridge is noisy and, on the evidence, sometimes
wrong.** S-A (decode the proposed block to text, re-encode with the target
tokenizer) introduces a real approximation the plan already priced in
(coarser acceptance, `docs/plans/.../section 3.5b`). But 50% exact-match
disagreement is a much stronger effect than "coarser acceptance," and we do
not have a confirmed explanation for it (decision register, X1 final
result). That gap has to be closed before trusting any downstream number,
because a method that silently produces wrong output some of the time is
not a method with a speed problem, it is a method with a correctness
problem.

## 2. Sequencing

Fix problem B's diagnosis before spending any GPU budget on problem A's
fix, because if X1's disagreement turns out to be a real bug rather than a
kernel-precision effect, refitting on top of it wastes the refit.

### Step 1 (diagnostic, cheap, first): root-cause the 50% disagreement

Run `native_ar` and `relay_p_cross_family` on the same single prompt with
`profile_regions: true` and full logit/state dumps enabled for the first
diverging cycle. Specifically:
- Log, every cycle, the target's own greedy argmax token at the anchor
  position under both code paths.
- The two paths must agree on this value up until the first commit that
  differs. Find the first cycle where they do not, and dump both paths'
  `target_cache` occupancy and the exact input fed to `native_target` that
  cycle.
- If the source of divergence is a real state bug (wrong cache crop length,
  wrong position id, a stale `conditioned_context`), it will show up as a
  structural difference in what was fed to the model, not just a
  near-tied logit. If it is genuinely a block-kernel-vs-single-token
  numerical effect, the two paths will show a near-tied logit gap (under
  1e-2 in log-probability) at the diverging position and nothing else
  different.

This needs no new training and one short interactive job (single GPU is
enough, this is not a throughput measurement). Budget: under 15 minutes of
GPU time.

**Decision gate:** if Step 1 finds a real state bug, fix it and re-measure
X1 as currently fit before touching the fitting objective, since the fix
might already close most of the gap. If Step 1 confirms it is a
kernel-precision effect (consistent with the rate being higher than
same-family only because coarser, block-variable-length verification
crosses more near-ties), proceed to Step 2 without expecting Step 1 to move
the acceptance-length number, which is governed by problem A, not problem
B.

### Step 2 (the main fix): give cross-family fitting real per-position supervision

Replace last-position-only supervision with a text-span alignment between
the two tokenizations of the same training example, so that most positions
carry a real target instead of being dropped.

Implementation, in `scripts/train_relay.py`'s `cross_family` branch:
1. For each fitting example, compute the character offset span of every
   source token and every target token in the shared underlying text
   (`tokenizer(..., return_offsets_mapping=True)` gives this directly for
   both HF tokenizers, no custom BPE logic needed).
2. For every target position, find the source position whose character
   span has the largest overlap with the target position's span. This is a
   direct, deterministic alignment, not a learned or DTW-based one: two
   spans over the same string either overlap or they do not, so "largest
   overlap" is well defined and requires no dynamic-programming machinery.
   Positions with no overlapping source span (rare, only at
   tokenizer-specific special-token boundaries) are dropped from the loss,
   same as today, instead of all-but-one position being dropped.
3. Gather `source_features` at the aligned source positions (one gather per
   example, an `index_select` along the sequence dimension) before the
   existing loss computation. Everything downstream of that point in
   `train_relay.py` is unchanged: same loss function, same relay
   architecture, same optimizer.

This is a data-alignment fix, not a new model. No new learned parameters,
consistent with the plan's stated preference for S-A over a costlier
learned bridge (Section 3.5b) as long as it can be made to work. Refit X1
with this objective and re-run the same 8-prompt dev benchmark used before.

**Kill criterion, stated in advance:** if aligned-position fitting still
leaves mean accepted tokens per cycle under 2.0 (only marginally above the
forced floor), the problem is not fitting-signal starvation and a linear
relay fit alone is not the bottleneck; stop here rather than escalating to
Step 4's learned bridge on a still-unexplained failure.

### Step 3 (the discrete-interface fix): cut how often re-tokenization drift happens

Independent of Step 2, implement S-B from the plan (vocabulary-intersection
map with S-A fallback, the plan's own stated refinement over pure S-A,
citing `timor2025heterogeneous`):
1. Build a static, one-time map: for every token in the source vocabulary
   whose decoded string form exactly matches a token in the target
   vocabulary, record the pair. This is a pure tokenizer inspection, no
   model or GPU needed, and can be computed and cached once per
   source/target tokenizer pair.
2. In `cross_family_relay_dflash_generate`, when re-encoding a decoded
   chunk, first check whether the chunk retokenizes identically through
   the intersection map; only fall back to full string re-encoding (the
   current S-A path, kept as-is) when it does not.
3. This reduces the number of re-tokenization round trips per cycle
   without changing the verification semantics: the target still verifies
   whatever text was actually proposed, exactly as today, just with fewer
   positions needing the expensive round trip.

Run independently of Step 2 first (it needs no refit, only a
`vocab_bridge.py` addition and a benchmark rerun on the existing X1
checkpoint), to measure its effect in isolation. Then combine with Step
2's refit for the final measurement.

### Step 4 (only if 2 and 3 together still fail the kill criterion): a small learned bridge

This is the "try neural nets" branch, held in reserve and only attempted if
Steps 2 and 3 together do not clear mean accepted tokens per cycle of
roughly 4 (representative of the weaker same-family cells, not the
strongest ones). Design, sized to stay small relative to the frozen
323M-parameter proposer trunk it sits next to:

- A low-rank output head `H = U V`, `U: 2560 -> r`, `V: r -> |V_target|`,
  with `r` on the order of 64 to 128, trained by distillation: feed the
  frozen proposer's own predicted continuation (in source vocabulary) into
  the aligned-position pipeline from Step 2, and train `H` to match the
  target's own next-token distribution at aligned positions, with the
  proposer and relay both frozen. Parameter count at `r=128`,
  `|V_target|~128000`, `2560` width is about 33M, an order of magnitude
  under the frozen proposer, avoiding the plan's own flagged risk
  (Section 3.5b, S-C) of the bridge becoming comparable in size to the
  model it is meant to be a thin adapter for.
- This replaces string re-encoding for the head of the distribution (where
  most probability mass sits) while keeping S-A as the fallback for
  anything the low-rank head does not cover, so correctness never regresses
  below what Step 3 already established.
- Explicit non-goal: this is not "train a new proposer." The 5-layer
  DFlash trunk stays completely frozen throughout; only `H` is trained,
  and it is trained once per source/target vocabulary pair, not per target
  size, so it amortizes across every future target in that family pair.

## 3. What "use all turing GPUs to parallelize" means concretely here

The account's real, verified quota (`sacctmgr show assoc`, decision
register) is 8 GPUs total for the whole lab, not per-job and not 88. Every
job in this project already uses exactly 4 (one full node), so the
achievable parallelism ceiling is **two concurrent 4-GPU jobs**, not more.
The plan uses that ceiling wherever two steps are truly independent:

- Step 1 (diagnostic) and Step 3's vocabulary-intersection map (pure CPU,
  tokenizer-only, no GPU at all) run concurrently with each other from the
  start, since neither depends on the other's output.
- Step 2's refit (once ready) can run concurrently with any repeat of Step
  1's diagnostic on a different prompt, if Step 1 needs a second pass.
- Step 4, if reached, is a single training run; nothing else in this plan
  is independent enough to pair with it, so it runs alone.

No step in this plan benefits from more than 4 GPUs each (all training and
generation here is single-node DDP, matching every prior job in this
project), so requesting more per job would not be used and is not planned.

## 4. What this plan does not attempt

It does not revisit X2 (already a strong, working, positive result) or the
already-closed E1/E4/E3/E6/E7/E8 lines. It does not attempt a full
from-scratch cross-family proposer (that is what DFlash/EAGLE-3 training
already is, and defeats the point of this paper). It stops at Step 4's
kill criterion rather than escalating indefinitely if a small learned
bridge also fails to clear a useful acceptance rate, in which case X1
remains reported exactly as it is now: a working, honestly negative
result, and the paper's claim stays that a fresh, same-tokenizer fit (X2)
is the recommended and evidenced route to a new family, with cross-tokenizer
retargeting stated as unresolved rather than solved.
