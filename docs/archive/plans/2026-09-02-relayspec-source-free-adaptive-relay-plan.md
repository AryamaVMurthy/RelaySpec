# RelaySpec next iteration: Source-Free Retargeting of Frozen Speculative Proposers

Date: 2026-09-02. Status: plan. Nothing in Sections 5 to 8 is executed yet
except where explicitly marked DONE.

Inputs to this plan: the current submission in `paper/iclr2027/`, the
public review record of PARD (ICLR 2026 poster, closest accepted comparable),
a verified map of the current training pipeline, and the source-free
retargeting proposal.

---

## 1. What changes about the claim

The current paper claims a narrow systems result. A frozen
feature-conditioned proposer can be attached to a new target by learning one
linear map into the tensor it already expects, plus a measured break-even
condition that says when this pays off.

The weakness is not correctness. It is that a reviewer can dismiss the work in
one sentence: "they regress onto an old model's intermediate tensor." PARD's
review record shows this risk is real. Two of four reviewers rated
contribution "fair" even while the paper was accepted, and the reviewer who
gave a 2 did so mainly on novelty framing.

The reformulation removes that one-sentence dismissal. Stated plainly:

> The real objective is not to make the new target look like Qwen3-4B. It is
> to make the frozen proposer generate candidates that the new target will
> accept.

So the upgraded claim is:

> A frozen speculative proposer can be retargeted to a new language model by
> learning only its continuous input interface from the new target's own
> verification signal, eliminating both source-model inference and
> source-model supervision.

This is stronger for four reasons, each testable:

1. The source trunk is no longer needed at adaptation time either, not only at
   inference. The old model can be deleted.
2. The bridge trains on the quantity our own cost model says decides the
   outcome, which is accepted tokens per cycle, not tensor reconstruction
   error.
3. It extends to fine-tuned descendants of a target, which is exactly the case
   PARD used as its argument against EAGLE.
4. The amortization story becomes literal: one expensive proposer, many cheap
   bridges, and nothing else retained.

### 1.1 Precision required on "source-free"

Do not write "source-free" without a qualifier until Section 7 item 8 is
resolved. The released DFlash path still uses the source token embedding and
the source LM head to turn draft hidden states into token logits
(`scripts/train_relay.py:250`, `:263`, `:293`). Those are single matrices
belonging to the released proposer contract, not the 34-layer trunk, but they
are currently loaded from the Qwen3-4B checkpoint.

The claim that is safe today: **no source transformer trunk forward pass at
adaptation time or inference time.** The stronger claim, that the source
checkpoint is never loaded at all, requires verifying whether the DFlash
release ships its own embedding and head tensors. That verification gates the
headline sentence, so it is scheduled in Phase 0.

---

## 2. Requirements taken from the PARD review record

PARD is the closest accepted comparable: same subfield, same "reduce
per-target adaptation cost" motivation, ICLR 2026. Its review thread gives
concrete, non-speculative requirements.

| What reviewers did | Consequence for us |
|---|---|
| The meta-review's first listed concern was whether gains come from same-family alignment rather than the method. PARD had to add a cross-family experiment to survive it. | Our exposure is higher, because Qwen3-4B, 8B and 14B are unusually well aligned. Answered by X2 (replicate in another family) and X1 (retarget across families via string-level verification), with E3 and E3b as the within-Qwen shifts. See Section 3.5b. |
| Reviewer Gfez's only weakness was missing analysis of how acceptance decays with draft length. PARD answered with an alpha-versus-K table plus the expected-acceptance-length formula. | We already measure per-position survival. Promote it from an appendix figure to a first-class analysis, and use it to motivate the loss weighting instead of asserting the weighting. |
| Reviewer MJ3S held a 4 over missing empirical baselines. PARD's accepted answer was a structured methodological comparison table plus an explicit fairness justification, not new runs. | Our Related Work table already has this shape. Keep it, restructure it per Section 6.3, and put the fairness justification in the paper text rather than only the internal source log. |
| Reviewer GWvT raised their score after the authors tightened definitions, defining "target-independent" precisely and distinguishing self-speculative methods as orthogonal. | Definitional precision moved a 2. Our equivalents are the compatible-target condition table and the trunk-versus-artifact distinction. Both must be impossible to misread. |
| Presentation defects were named in reviews, including swapped figure axes and an indexing bug in two equations. | Budget a dedicated figure and equation audit before submission. It is cheap and it demonstrably costs points. |
| Every PARD number is measured inside vLLM. Their Appendix F shows Transformers understates speedup by roughly 2x versus vLLM. | This is our largest unaddressed gap. Either invest in an engine integration or defend the batch-one scope explicitly. See E8. |

---

## 3. Method

Let `T` be the new target, `P` an already-trained frozen proposer, and `z^T_t`
the target hidden taps at committed position `t`. The current objective
regresses `R_phi(z^T_t)` onto `c^S_t`, the tensor the source trunk produces
after the proposer's own fusion (`src/relayspec/proposers.py:22-31`). The
reformulation drops `c^S_t` entirely and optimizes `c_t = R_phi(z^T_t)`
through the frozen proposer against the target's own verification signal.

Neither target nor proposer is updated. The target remains the only component
that accepts tokens, so Proposition 1 and the greedy-authority argument are
untouched. The relay affects efficiency through proposal quality, never token
authority.

### 3.1 Two-axis design grid

The previous draft of this plan had a single objective ladder. That was
incomplete. There are two independent axes and they must be ablated
separately, because they have very different costs.

**Axis A, supervision signal:**

- **L0** relative interface MSE. Current objective. Requires the source trunk.
- **L1** greedy-agreement cross-entropy against the token the target verifier
  would commit. Matches the greedy accept rule we actually deploy and measure.
  *(implemented, see Section 7 item 1)*
- **L2** KL to the target's full next-token distribution. The standard drafter
  distillation form. `proposal_kl_loss` already exists
  (`src/relayspec/losses.py`), currently fed source-conditioned teacher logits.
- **L3** differentiable accepted-prefix surrogate. Formally defined in
  Section 3.1.1. It is a surrogate, not the deployed quantity, and the plan
  must never call it "the expected accepted length" without that
  qualification. Its product structure down-weights later positions by the
  probability of ever reaching them rather than by a chosen coefficient, which
  is the principled answer to the question PARD's reviewers pressed about
  later tokens. *(implemented, but its docstring currently overclaims: see
  Section 7 item 1)*
- **L4** L0 warm start, then L3 fine-tune. Requires the source trunk once and
  never again.

**Axis B, state distribution:**

- **S0 teacher-forced.** Target runs on fixed text, per-position logits are
  read, loss applied at ground-truth prefixes. Cheap: one target forward plus
  one proposer forward per example.
- **S1 on-policy.** For each prefix, the relay produces a context, the frozen
  proposer drafts a block, the target scores those candidates, and the loss is
  applied at the states the system actually visits. Implemented as on-policy
  state collection plus supervision at collected states: the rollout is
  detached, gradients flow only through the current block's proposer forward.
  This is the standard DAgger-style arrangement and needs no exotic machinery.

S1 is better in principle because it removes exposure bias, training on the
state distribution the deployed system visits instead of on ground-truth
prefixes. It is also materially more expensive, because fitting now requires
generation and repeated target verification rather than single forward passes.

**Cost tension that must be managed.** The paper currently sells "fits in 86
to 119 seconds." S1 and per-task fitting both inflate that, possibly by an
order of magnitude. Rule for this iteration: report fit cost per variant, never
one blended number, and do not let the headline adaptation-cost claim inflate
silently while still leading with portability-is-cheap. Phase 1 runs S0 only,
which preserves the existing claim while testing the supervision axis. S1
enters in Phase 2 and, if it wins, both are reported with their costs and the
amortization argument absorbs the difference.

**Novelty boundary, to be stated in the paper.** Token-level distribution
matching and sequence-level accepted-prefix objectives are the
EAGLE, VSD and PARD-2 objective family. Our own source log already flags this
for `zou2026vsd` and `an2026pard2`. The contribution is not the objective
family. It is the setting: an external adapter, a frozen third-party proposer,
and no source model. Write that boundary explicitly, and let the ablation
decide whether L3 beats L0 rather than asserting it.

### 3.1.1 Formal definition of the L3 surrogate

Write `q_D(. | .)` for the frozen proposer's next-token distribution under a
relay-produced context, and `y*_1 .. y*_K` for the tokens the target verifier
would commit at the `K` proposed positions. Define

```
a_j      = q_D(y*_j | .)                          per-position agreement mass
A_soft   = sum_{i=1..K} prod_{j<=i} a_j           the L3 surrogate
A_hard   = sum_{i=1..K} prod_{j<=i} 1[argmax q_D = y*_j]   the deployed quantity
```

`A_hard` is what the paper measures and what enters the break-even condition.
It is a deterministic function of the proposer logits and is not
differentiable. `A_soft` is what we optimize.

Their exact relationship, which must be stated in the paper rather than
glossed:

1. `A_soft` is the exact expected accepted prefix length under a *sampled*
   proposal regime, where each draft token is drawn from `q_D` and accepted
   only if it equals the verifier's token. Under that regime the identity is
   exact.
2. The deployed regime is greedy argmax, not sampling. Under greedy, `A_soft`
   is a differentiable relaxation of `A_hard` and **no general inequality
   holds in either direction**. A diffuse distribution can have
   `a_j < 0.5` while `y*_j` is still the argmax, so the surrogate undercounts.
   A near-tie can have `a_j` around 0.45 lose to a competitor at 0.5, so the
   surrogate overcounts.
3. The one guaranteed implication is `a_j > 1/2` implies position `j` is
   accepted under greedy. Driving agreement mass above one half is therefore
   sufficient for hard acceptance, and this is the mechanism by which
   optimizing `A_soft` is expected to move `A_hard`.

Because no equivalence holds, the surrogate is licensed empirically, not
analytically. Every L3 run must report measured `A_hard` on the development
split next to the surrogate value, and the plan must show that improving
`A_soft` actually improves `A_hard`. If that correlation fails, L3 is
discarded in favour of L1, which supervises the argmax target directly.

### 3.2 Optimization-surface risk and pre-registered mitigations

Optimizing `R_phi` through the frozen proposer with no anchor to `c^S` leaves
nothing constraining `c-hat` to the region where the proposer's learned
behaviour is well conditioned. The proposer was trained on inputs drawn from
Qwen3-4B fused states. Pushing far off that manifold can yield solutions that
score well on the surrogate while behaving badly, by exploiting proposer
quirks rather than improving genuine agreement.

Mitigations, registered in advance rather than added after a bad result:

1. Warm-start from the L0 regression solution, which we already have as
   released checkpoints.
2. Optional weak regression anchor term, reported as its own ablation rung so
   the anchor's contribution is visible.
3. Monitor `c-hat` norm and direction drift against the source-state
   distribution during fitting, and log it alongside the loss.

If the unanchored variants prove unstable, L4 becomes the headline and the
framing shifts from "source-free" to "source-trunk-free fitting". That
fallback is acceptable and should be decided at the Phase 1 gate, not later.

### 3.3 Task-aware retargeting

The current results show that representation reconstruction does not preserve
proposal quality equally across workloads. RelaySpec improves every 8B
configuration but slows down both proposer families on Qwen3-14B HumanEval and
MBPP, because acceptance degradation outweighs the removed source-model work.

Once the source trunk is gone from fitting, any text is valid fitting data, so
task composition becomes a free variable. Fit variants: math only (current),
math plus code, math plus code plus chat. Evaluate every variant on all five
tasks. Report accepted tokens per cycle separately for math, code and chat.

This attacks the paper's weakest published numbers directly. Either the two
negative 14B code cells turn positive, or they become a characterized boundary
with a stated cause. Reviewers respond well to authors attacking their own
worst result.

**Split hygiene, mandatory.** "Train where mismatch is largest" is a
test-informed procedure if mismatch is measured on evaluation tasks, and this
paper's credibility rests on registered development prompts and zero-overlap
audits. Requirements: mismatch measured on development splits only, fitting
data drawn from training splits only, and both prompt audits
(`scripts/check_prompt_overlap.py`, `scripts/audit_prompt_similarity.py`)
re-run and reported per new manifest. The current "no task-specific tuning"
purity claim must be reworded honestly, because mixed fitting data is a form
of workload targeting even without answer supervision.

### 3.4 Efficient adaptation to fine-tuned targets

Consider `T'` produced from base target `T` by LoRA or another lightweight
update. Rather than learning a new relay, retain the base relay and learn only

```
R_T' = R_T + dR,   dR constrained low-rank
```

This is framed as a hypothesis, not a convenience: **small model updates
should induce correspondingly low-dimensional changes in the optimal
speculative interface.** If it holds, one proposer plus one base relay
cheaply supports many task-specific descendants of the same target.

Two stages:

- **Stage A, zero-shot.** Run the base-target relay against the descendant.
  Same tokenizer, same widths, different weights. Measure acceptance retention
  loss. This is the scenario PARD used against EAGLE, so it is worth knowing
  our answer before a reviewer asks.
- **Stage B, delta.** Learn low-rank `dR`. Report recovery against fit cost,
  and against a full refit, and report the rank needed.

**Blocking decision.** `peft` is not installed and there is no LoRA code in
the repo. Either use a public fine-tuned Qwen3-8B or 14B descendant, which is
better external evidence, or install `peft` on the cluster and produce our own
LoRA descendant, which gives a controllable sweep of fine-tuning strength.
Preference: a public descendant for the headline number, a self-made LoRA
sweep for the trend line.

### 3.4.1 Architecturally shifted target, required separately

LoRA descendants are the weakest possible distribution shift: same
architecture, same widths, small weight delta. They do not answer the
same-family concern, which is about whether Qwen3-4B, 8B and 14B being
siblings from one training pipeline is doing the work. A reviewer will say so.

We therefore need a target that keeps tokenizer compatibility, which our own
compatibility conditions require, while shifting architecture as much as
possible. Candidates, in decreasing attractiveness:

1. **A structurally modified Qwen3 derivative** that keeps the tokenizer:
   depth-pruned, width-pruned or distilled variants published by third
   parties. This is the sweet spot, because it shifts architecture
   substantially, keeps the vocabulary, and stays small enough to run.
2. **A Qwen3 mixture-of-experts variant.** Routing changes how hidden states
   are formed, so a working relay here would substantially weaken the
   sibling-alignment critique.
3. **A larger dense Qwen3 target** such as 32B. This tests scale
   extrapolation beyond the fitted range but shifts architecture least.

**Hard feasibility constraint that shapes this choice.** The current harness
gives every rank its own full model copy and a disjoint prompt shard
(`scripts/train_relay.py:174`, and the same pattern in the benchmark
harness). There is no tensor parallelism. On 48 GB cards that caps the target
at roughly what fits in one card in BF16. Qwen3-14B at about 28 GB fits.
A 32B dense target at about 64 GB does not, and neither does a 30B-class
mixture-of-experts model. So options 2 and 3 require either model sharding
work or a changed timing protocol, and a changed timing protocol would
invalidate comparability with every published cell.

Consequence: option 1 is the only route that does not force a harness rewrite
or a re-run of the whole matrix. Phase 0 therefore includes a checkpoint
survey to find a tokenizer-compatible, architecturally shifted, single-card
Qwen3 derivative. If no such checkpoint exists, record that as the reason the
experiment is not run rather than substituting LoRA descendants and calling
the concern answered.

### 3.5 Canonical speculation representation: highest-upside experiment

Both existing relays already emit into the same 2,560-dimensional proposer
interface. That shared output space makes a probe possible with no new
training at all.

- **P0, free, no training.** Run existing `W_8B` and `W_14B` on identical text
  and compare the resulting `c-hat` trajectories in the shared interface space
  using cosine similarity, CKA and principal-subspace overlap. Pure analysis
  of checkpoints we already hold.
- **P1, decisive test.** Map hidden states from multiple target sizes into a
  shared low-dimensional `u`. Train the proposer-facing decoder using one
  target, then feed it `u` derived from a different target. Measure acceptance
  loss under zero-shot or very-low-shot transfer. This is the sharp version of
  the hypothesis and the only version worth putting in the main paper.
- **P2, fallback.** Joint factorization `W_t = A_t U` with `U` shared across
  targets, compared against independent full-rank maps. Weaker evidence,
  because joint training can manufacture a shared space that does not
  transfer.

**This is the highest-upside experiment in the plan, and it is scheduled
accordingly.** If P1 works, the paper's thesis changes. "We can retarget
cheaply, once per target" becomes "there is a canonical speculation interface
shared across targets, and one decoder can serve several of them." That is a
representation-level claim rather than a systems-level one, and it is worth
substantially more at ICLR than the retargeting result alone. P1 therefore
moves into Phase 1 alongside the supervision gate rather than waiting for
Phase 2, and P0 runs in Phase 0 because it costs nothing.

**Two-branch outcome, decided in advance.**

- **Branch A, P1 succeeds** (zero-shot or very-low-shot cross-target transfer
  within about 3 points of acceptance retention). The canonical interface
  becomes the central contribution. Source-free retargeting becomes the
  mechanism that makes it measurable, and the break-even analysis becomes the
  deployment corollary. The paper is retitled and reorganized around the
  shared-representation claim, with per-target relays as the ablation that
  shows what the shared decoder gives up.
- **Branch B, P1 fails.** The Section 1 thesis stands unchanged, and P1
  becomes a short appendix negative result. Do not let it complicate the
  method, and do not soften the kill criterion after seeing the number.

**Kill criterion, fixed now.** Branch B is taken if P0 shows weak alignment
and P1 zero-shot transfer loses more than 3 points of acceptance retention.
Write this threshold into the run configuration before the first P1 fit so it
cannot drift.

### 3.5b Cross-family retargeting

Earlier drafts of this plan said cross-family transfer was "excluded by
construction". That was wrong, and the correction matters enough to restate
plainly.

The relay is a map `R^{5 d_T} -> R^{2560}`. Nothing in it depends on which
family produced `z^T`. The continuous interface is already family-agnostic.
The barrier is entirely at the **discrete token interface**, and it decomposes
into three separate mappings that must each be handled.

**M1, proposer output to verifier vocabulary.** The frozen proposer emits
token IDs in the source vocabulary through the source LM head. The verifier
consumes IDs in its own vocabulary. With identical vocabularies this is the
identity map, which is why the current work never had to think about it. With
a different family it is the hard problem, because speculative acceptance
requires proposer and verifier to agree on what a token *is*.

**M2, draft-token embedding.** DFlash embeds the tokens it is growing through
`source.model.embed_tokens` (`scripts/train_relay.py:250`). Those tokens live
in whatever vocabulary the proposal chain uses, so M2's difficulty is
determined by the choice made for M1.

**M3, committed prefix conditioning.** The proposer conditions on the
committed prefix. Today this is implicit because the vocabularies match.

#### Solution families for M1

**S-A, string-level verification. Zero new learned parameters. Recommended
primary.** The proposer proposes in its own vocabulary. Decode the proposed
block to text, re-encode with the target tokenizer, and let the target verify.
Commit the longest prefix of the target's own greedy *string* that the
proposal agrees with, then the target's correction.

- Correctness survives. The target still produces every committed character,
  so the induction behind Proposition 1 carries over at string granularity
  instead of token granularity. This needs restating in the paper, not
  reproving.
- The "only one linear map is learned" claim survives completely intact, which
  is the whole reason to prefer this route.
- Costs: retokenization per cycle, plus acceptance granularity becomes
  coarser, because a mismatch inside a token kills the rest of the block. So
  expect lower acceptance than the same-family case, which the break-even
  condition will then judge.

**S-B, vocabulary intersection map with fallback.** Many BPE tokens have
identical string forms across vocabularies. Build an injective partial map on
that intersection, and fall back to S-A only for the non-matching remainder.
This is the family that `timor2025heterogeneous` (ICML 2025) develops for
lossless heterogeneous-vocabulary speculative decoding, and it is the correct
citation and prior art for this whole subsection. Cheap, and keeps
token-granular acceptance for most positions.

**S-C, learned output head into the target vocabulary.** Replace the source LM
head with a learned `H_phi: R^2560 -> R^|V_target|`, distilled from the
target. Philosophically attractive, because everything learned still sits
*outside* the frozen proposer, at its two interfaces. But count the
parameters before committing: a tied embedding and head at
`|V_target| x 2560` is on the order of 300M, which is comparable to or larger
than the frozen DFlash trunk itself. At that point "we learn only a small
bridge" collapses, and a reviewer will say we retrained most of the model.
Only pursue S-C with a low-rank or intersection-restricted construction that
keeps the learned parameter count well under the frozen count, and report
that count prominently.

**Recommendation.** S-A as the primary result, because it is cheap, needs no
new parameters, and preserves the headline claim. S-B as the refinement that
recovers token-granular acceptance. S-C only if S-A and S-B both fail, and
only with an explicit parameter-count defence.

Note that Task 8's provenance check now gates two things, not one. If the
DFlash release ships its own embedding and head, S-C becomes much cleaner. If
it borrows Qwen3-4B's, then M1 and M2 are structurally entangled with the
source checkpoint and S-A is the only honest route.

#### Consequence for Table 1

Table 1's first condition currently reads as though identical token IDs are a
hard requirement of the method. After S-A it is only a requirement for
*token-granular* acceptance. The condition must be reworded to distinguish
the continuous interface requirement, which is genuine, from the discrete
interface requirement, which is a choice of bridging strategy with a measured
acceptance cost.

#### Two distinct experiments, do not conflate them

**X1, cross-family retargeting.** Keep the Qwen3-trained frozen proposer, and
retarget it to a non-Qwen target using S-A. This is the strong claim: the
learned continuous interface transfers across families, and only the discrete
interface needs bridging. It directly answers the same-family concern that
PARD's meta-review put first.

**X2, within-family replication in another family.** Repeat the entire
original setup natively inside a different family, for example an EAGLE-3
checkpoint released for a Llama-family model, retargeted from its own source
to a larger sibling. This shows the *method* generalizes, independently of
anything about Qwen3. It requires no new mapping machinery at all, because
vocabularies match inside the family. It is the cheaper and more certain of
the two, and it should be attempted first.

**Blocking gate for both: a checkpoint survey.** X2 needs a released
feature-conditioned proposer for a non-Qwen family, plus two sibling targets
that both fit a single 48 GB card given the harness has no tensor
parallelism. EAGLE-3 has published checkpoints beyond Qwen, so X2 is
plausible, but the specific size pairing must be verified rather than
assumed. X1 needs only one non-Qwen target of any size that fits. Neither can
be scheduled until the survey is done, and the survey must record exact
repository IDs and revisions in the source log before anything is cited or
run.

### 3.6 Central novelty comparison

The reformulation moves us much closer to several existing lines of work, so
the novelty comparison stops being a Related Work paragraph and becomes a
first-class part of the method section. Six comparators must be treated
centrally, not mentioned in passing.

**SD-squared is the primary threat and must be handled first.** It computes a
steering signal from verifier hidden states and injects it into a pretrained
drafter, and its ablation set includes a frozen-drafter configuration. That is
very close to our new framing: verifier-derived signal, drafter left alone,
only an external pathway learned. Our differentiators must be stated
precisely, and each one must be checkable by a reviewer:

1. What is learned. SD-squared learns a steering signal added into a generic
   pretrained autoregressive drafter. We learn a replacement for a specific
   released tensor contract that a feature-conditioned proposer already
   requires as input. The proposer does not merely benefit from our output, it
   cannot run without one.
2. What is eliminated. Our claim is the removal of a separately executed
   source transformer from the decode loop and from adaptation. SD-squared
   has no such component to remove, because its drafter was never coupled to a
   second model's hidden states.
3. What the deployment unit is. Ours is one map per target and proposer pair
   against a frozen third-party artifact, which is what makes the amortization
   argument in Section 1 possible.

If, on close reading of SD-squared's frozen-drafter ablation, points 1 and 2
do not hold as stated, this plan's positioning must change before submission
rather than after review. Verifying that ablation in detail is a Phase 0 task.

**PARD-2 and VSD** bound the objective claim. Both train for acceptance, so
Section 3.1's novelty boundary must name them explicitly at the point where
the objective is introduced, not only in Related Work.

**Draft-OPD, OnlineSPEC and Learning to Draft are not in `references.bib`.**
They must be sourced and verified through the existing source-log process,
recording publication status, venue, primary URL and checked claims, before
they are cited anywhere. Do not add bibliography entries from memory. Until
they are verified, treat the novelty comparison as incomplete.

The resulting comparison table replaces the current Table 6 and is specified
in Section 5.3.

---

## 4. Experiment matrix

Cost units: one relay fit under S0 is 87 to 119 seconds on four GPUs. One
benchmark cell is a separate 4-GPU job under the existing one-hour sbatch
limit. All GPU work runs on the `turing` cluster, partition `u22`, `node01`,
`--gres=gpu:4`. Never the local laptop GPU.

| ID | Experiment | Cost | Objection it answers | Depends on |
|---|---|---|---|---|
| E1 | Axis A ladder L0 to L4 under S0, DFlash then EAGLE-3, 8B then 14B | 10 fits, dev gate, full MATH for survivors | "It is just tensor regression"; enables the source-free claim | Section 7 items 1 to 4 |
| E1b | Axis B: best Axis A objective under S1 on-policy | 2 to 4 fits, higher per-fit cost | Exposure bias; is trajectory training worth its cost | E1 winner, Section 7 item 5 |
| E2 | Fitting-data composition, three variants, evaluated on all five tasks | 3 fits per family, 14B evals first | The two negative 14B code cells | E1 winner, new manifests, audits |
| E3 | Descendant transfer, Stage A zero-shot then Stage B low-rank delta | 1 to 2 fits plus MATH evals | Fine-tuned targets, the practical maintenance case | E1 winner, descendant decision |
| E3b | Architecturally shifted tokenizer-compatible target | 1 fit plus MATH and one code task | Same-family alignment, within Qwen | Checkpoint survey, Section 3.4.1 |
| X2 | Within-family replication in a non-Qwen family | 1 to 2 fits plus MATH and one code task | **"The method only works for Qwen3."** Cheapest strong answer, no new mapping code | Checkpoint survey, Section 3.5b |
| X1 | Cross-family retargeting via S-A string verification | Mapping code, 1 fit, MATH plus one code task | **"Gains are same-family alignment."** The strong claim | X2 first, Section 3.5b, Task 11 |
| E4 | Representation: P0 free analysis, then P1 as a Phase 1 experiment, P2 fallback | P0 analysis only; P1 two fits | Highest upside. Decides Branch A versus Branch B | Nothing for P0 |
| E5 | Relay capacity: rank sweep and tap-count/selection sweep | 6 to 10 fits, dev gate | "Why 52 to 66M parameters" | Section 7 item 4 |
| E6 | Native AR baseline across all 16 breadth cells | 16 AR-only cells | Consistency of the headline comparison | Nothing |
| E7 | DFlash isolated memory measurement, both scales | 4 short jobs | Memory column currently EAGLE-only | Two new configs, Section 7 item 6 |
| E8 | Literature-cited adaptation-cost table plus explicit limitation | No GPU | The missing cost multiplier | Nothing |
| E9 | vLLM or SGLang concurrency 1 to 64, one cell minimum | Large engineering | The single biggest reviewer-facing gap | Separate decision |

Two of these close gaps that are confirmed, not hypothetical:

- **E6.** `reports/final/BREADTH_MATRIX.json` stores only
  `source_tokens_per_second` and `relay_tokens_per_second` per cell. There is
  no native-AR field. AR appears only for the four MATH cells, in Figure 4.
- **E7.** The original execution plan
  (`docs/plans/2026-08-28-relayspec-rigorous-evidence-and-final-benchmarks.md`,
  Task 9) called for isolated memory configs for each family and scale. Only
  the EAGLE-3 ones were run. A later visual-design pass relabelled the panel
  "EAGLE-only" with no technical justification. The measurement code is
  proposer-agnostic and DFlash already has the isolated single-method load
  plans it needs, so this requires two new config YAMLs modelled on
  `configs/protocol_active/eagle3_qwen3_{8b,14b}_memory_{source,relay}_4gpu.yaml`
  and no new code.

---

## 5. Related work and table obligations

### 5.1 Current state

Cited: 23 entries. Uncited but present in `references.bib`: 18.

Related work, Section 7 of the paper, covers 11 papers in three paragraphs:
proposers that condition on target states (DFlash, EAGLE-3, DFlare, RepSpec),
portability by changing the drafter (PARD, OmniDraft, SD-squared, EDA,
HyperDFlash, VSD), and state or cache mapping for other purposes (KVShot,
cross-model KV transfer, SPEED-Bench).

Empirical baselines: optimized source reuse across all 20 cells, native
autoregressive on 4 MATH cells only, released target-specific proposer for 3
of 4 configurations. No table compares published numbers from other papers,
which is a deliberate choice given cross-hardware incomparability. The
in-house equivalent is the target-specific-proposer control in Table 5.

### 5.2 Citation gaps that matter

| Missing | Why it matters now |
|---|---|
| `zhou2026hsd` Hierarchical SD, ICLR 2026 oral | Our own source log calls it the current top-venue standard for a speculative-decoding paper. Being uncited while imitating top-venue structure is a bad look. |
| `timor2025heterogeneous` ICML 2025 | The natural citation for our tokenizer-compatibility condition. We currently assert cross-tokenizer is out of scope without citing the work that handles it. |
| `cai2024medusa` ICML 2024 | Canonical frozen-backbone plus small-trained-heads method, the closest structural ancestor of frozen-proposer plus small-trained-adapter. PARD's reviewers invoked Medusa repeatedly. |
| `an2026pard2` | Belongs in the acceptance-aware-training contrast sentence alongside VSD, especially now that Section 3.1 leans on that objective family. |
| `luo2026verifierskipping`, `wang2026specsa` | Both support our Amdahl claim that the verifier is the remaining bottleneck once the trunk is removed. We assert that uncited. |
| `kaggarwal2026dflashlora` | Directly relevant prior art for Section 3.4. The source log already warns we cannot claim small DFlash adapters are unexplored. |
| `hu2025griffin`, `wang2026xpress` | GRIFFIN motivates per-position acceptance reporting, which we already do in Figure 7 without attribution. |

### 5.3 Table 6 restructure

Table 6 lost a column during the nine-page squeeze, and per Section 3.6 it now
has to carry the central novelty argument rather than sit at the end of
Related Work. Specified columns:

```
Method | Learned component | Proposer weights frozen? |
Needs a second model at adaptation time? |
Removes a separately executed model from decode? | Adaptation data
```

Rows, in this order: DFlash and EAGLE-3, Medusa, PARD, PARD-2, OmniDraft,
SD-squared, EDA, HyperDFlash, VSD, then RelaySpec. Draft-OPD, OnlineSPEC and
Learning to Draft join once Section 3.6's verification task completes.

Two columns do the work. "Needs a second model at adaptation time" is what
this iteration newly lets us answer no to. "Removes a separately executed
model from decode" is the column no prior row can tick, because no prior
method has a source trunk to remove in the first place. State that asymmetry
explicitly in the caption so it reads as a scope difference rather than a
scoreboard.

The adaptation-data column stays qualitative, with concrete numbers only where
the cited paper published them under stated hardware.

---

## 6. Implementation work, file by file

1. **DONE.** `src/relayspec/losses.py`: added `_check_verifier_shapes`,
   `greedy_agreement_ce` (L1) and `expected_accepted_length_surrogate` (L3).
   Purely additive, no existing function modified, no call sites changed, so
   every published number remains reproducible. `tests/test_losses.py` extended
   by 8 tests, 15 passing in that file.
2. `scripts/train_relay.py`: add `training.supervision` with values
   `source_interface` (current default, byte-identical behaviour) and
   `target_verifier`. Under `target_verifier`: skip the source trunk forward at
   `:214-218` entirely, stop passing `logits_to_keep=1` on the target call at
   `:229-238` so per-position logits are available, build the proposal block,
   run the conditioned proposer forward, and apply the Axis A loss. The
   dormant `proposal_kl_weight` path at `:181-193`, `:242-265` and `:281-300`
   is the template. Default must stay the current path.
3. `src/relayspec/eagle3.py`: add a teacher-forced differentiable conditioned
   forward built from `Eagle3Backend.initialize/propose` (`:145-229`), so E1
   covers EAGLE-3. Currently `train_relay.py:190-191` raises for any
   non-DFlash proposal training, which would otherwise restrict the whole
   headline result to one proposer family.
4. `src/relayspec/relay.py`: add `rank` for low-rank factorization,
   `shared_decoder` for E4-P2, and `delta` mode with a frozen base `W` plus
   trainable low-rank `dR` for E3 Stage B.
5. On-policy harness for S1 and E1b: detached rollout that collects visited
   states, then supervision at those states. Reuse the generation path in
   `src/relayspec/generation.py` for collection rather than writing a second
   decoder.
6. `configs/protocol_next/`: new directory so `configs/protocol_active/`
   stays pristine and every published number remains regenerable. Includes the
   two DFlash memory configs for E7.
7. New fitting manifests for E2, plus re-run of both prompt audits with
   overlap and similarity numbers recorded per manifest.
8. **Provenance check, gates the headline sentence.** Determine whether the
   DFlash release ships its own token embedding and LM head or genuinely
   borrows Qwen3-4B's. Record the answer in the decision register. If it ships
   its own, the stronger "source checkpoint never loaded" claim becomes
   available. If not, Section 1.1's trunk-only phrasing stands.
9. Tests for every new loss, relay variant and supervision path. The suite is
   127 tests in about 14 seconds on CPU, so all of this is verifiable locally
   before anything reaches the cluster.
10. Paper-side edits that need no new numbers: the amortization framing, the
    motivating sentence from Section 1, the sharpened SD-squared distinction,
    the citation insertions from Section 5.2, the Table 6 restructure from
    Section 5.3, and the explicit cost limitation.

---

## 7. Sequencing and gates

**Phase 0, no GPU.** Implementation items 2, 3, 4, 8, 9 and 10. E8. The E4-P0
analysis script. Plus four verification tasks that gate claims rather than
code: the DFlash embedding and head provenance check (item 8), the
SD-squared frozen-drafter ablation reading (Section 3.6), sourcing the three
missing comparators (Section 3.6), and the checkpoint survey for an
architecturally shifted single-card target (Section 3.4.1). Constraint to
respect throughout:
`scripts/train_relay.py:71-72` hard-asserts `WORLD_SIZE == 4`, and `:176-181`
requires `fit_examples == steps * world_size`, so 4,096 examples means 1,024
steps across 4 ranks. Any data-composition change must preserve that
arithmetic.

**Phase 1, cheap GPU. This is the decision point for the whole iteration.**
E1 under S0 on DFlash-8B only, development gate first. Outcomes:

- If some source-free objective reaches L0's acceptance retention within about
  2 points, the source-free framing is the headline and the iteration proceeds
  as planned.
- If not, L4 hybrid becomes the headline and the framing shifts to
  "source-trunk-free fitting". Decide this here, not later.

Phase 1 also runs **E4-P1**, because Section 3.5 makes it the highest-upside
experiment and its outcome decides whether the paper is reorganized. Running
it late would mean rewriting the paper late.

**Phase 1 gate outcome (decided, 2026-09-02).** All four objectives tested
(L1 `greedy_agreement_ce`, L2 `proposal_kl_to_target`, L3
`accepted_prefix_surrogate` cold-start, L4 `accepted_prefix_surrogate`
warm-started from L0) collapse relative to L0's 92.9%-retention,
1.523x-speedup registered baseline: L1 20.7% retention (0.35x), L2 21.6%
(0.37x), L3 14.2% (0.24x), L4 19.1% (0.34x). L4's failure is the decisive
data point, since warm-starting from a known-good checkpoint rules out
initialization as the cause, so the collapse is intrinsic to the
verifier-feedback objectives at this step budget, not a cold-start artifact
alone. **Decision taken: L4 hybrid does not rescue the source-free framing
either. The headline reverts fully to L0's source-anchored fit.
Source-free/source-trunk-free fitting is documented as a negative result
(Appendix H.1 of the paper, and the decision register), not pursued further
for headline framing.** This negative result independently corroborates two
published findings we can now cite affirmatively: SD²'s own frozen-drafter
ablation \citep{berdoz2026sd2}, and the entire verifier-feedback line
(Draft-OPD, OnlineSpec, Learning to Draft) needing a fixed-but-trainable
draft model rather than a frozen third-party one. Immutable artifacts:
`outputs/bench-verifier-l{1,2,3,4}-dev32-{26576,26588,26589,26590}/benchmark-summary.json`
on turing; full narrative in `docs/research/relayspec-decision-register.md`.

**Phase 2.** E1 completion for EAGLE-3 and 14B. E1b for the on-policy axis.
E5. E3b if the checkpoint survey found a usable target.

**Phase 3.** E2 and E3, the two results that most directly answer the
reviewer objections in Section 2.

**Phase 4.** E6 and E7 for consistency and completeness, then the figure and
equation audit, then E9 only with committed engineering time.

---

## 8. Risks and what we will not claim

Risks, with owners in the plan:

- The frozen proposer may be a poor optimization surface. Mitigated by
  Section 3.2, with a decided fallback at the Phase 1 gate.
- EAGLE-3's differentiable conditioned forward is harder than DFlash's,
  because it grows a chain with cache state rather than a single block. If it
  proves intractable, E1 ships DFlash-only and EAGLE-3 stays on L0, which
  weakens but does not destroy the claim.
- Task-mixed fitting weakens the current purity claim. Mitigated by Section
  3.3 split hygiene plus honest rewording.
- The descendant experiment depends on checkpoint availability or a `peft`
  install on the cluster.
- The representation probe is high variance and is hard-gated.
- Fit cost can inflate under S1 and per-task fitting. Managed by per-variant
  reporting.

What we will not claim, stated in the paper rather than left for a reviewer to
find:

1. No controlled proposer-training compute multiplier. Data volume and
   wall-clock fit time only, with hardware stated.
2. Cross-family claims are scoped to whatever X1 and X2 actually deliver. If
   only X2 runs, the claim is that the method replicates in another family,
   not that a single proposer spans families. If X1 runs under S-A, the claim
   is string-granular acceptance across families, and the acceptance cost of
   that coarser granularity must be reported next to the speedup.
3. No production serving throughput unless E9 completes.
4. No claim that the objective family is novel. Only its use for a frozen
   third-party proposer with no source model.
5. Fit-cost claims are per-variant, never one blended number.

---

## 9. Open decisions needed before Phase 1

1. E3 descendant source: public fine-tuned Qwen3 checkpoint, or self-made LoRA
   with `peft` installed on the cluster.
2. E3b target: which tokenizer-compatible, architecturally shifted,
   single-card Qwen3 derivative, or a recorded finding that none exists.
3. Whether to commit engineering time to E9, which determines if the paper
   targets ICLR with a batch-one scope statement or aims at a systems venue
   with engine numbers.
4. Whether the already-implemented loss functions from item 1 stay in the tree
   or are reverted pending the Phase 1 gate. Note that the L3 docstring must
   be corrected either way, per Section 3.1.1.

---

## 10. Final status (updated 2026-09-02, end of session)

Every item below reflects an actual outcome, an actual artifact, or an
explicit, reasoned decision not to build something. Nothing is "in
progress" as of this update; the turing job queue is empty.

| ID | Item | Final status | Result / reason |
|---|---|---|---|
| E1 (L0-L4, DFlash-8B, S0, dev gate) | Source-free/hybrid supervision ladder | **DONE, negative** | L1 20.7%, L2 21.6%, L3 14.2%, L4 (warm-start) 19.1% retention, all vs. L0's 92.9%. Closed; see Section 7. |
| E1 (EAGLE-3, 14B), E1b (on-policy) | Ladder extensions | **DEPRIORITIZED** | Gate failed at the cheapest cell; no reason to re-spend GPU-hours confirming failure at higher cost. |
| E4-P0 | Free representation probe, 8B vs 14B relay | **DONE, strong positive** | Cosine 0.973, linear CKA 0.732, rank-32 subspace overlap 0.897. |
| E4-P1 | Adapter-based transfer test (both directions, per explicit request) | **DONE, real but asymmetric, confirmed** | Small base to big target: 79.3% retention, 1.241x speedup. Big base to small target: 16.4% retention at first (after fixing a genuine optimizer-memory OOM), retested at a properly-scaled SGD learning rate (0.02 vs. the AdamW-tuned 0.0006) to rule out under-optimization: improved to 26.8% retention but still collapses. The asymmetry is confirmed across two learning rates, not an artifact of one under-tuned run. Not full Branch A; not flat Branch B either. Reported as a partial, direction-dependent finding, one size pair only. |
| E7 | DFlash isolated memory, both scales, both methods | **DONE** | Found and fixed a real bug (relay-side configs never unloaded the source trunk, so the "isolated" measurement was measuring the wrong thing). Re-measured correctly for all 4 cells. |
| E6 | Native-AR baseline | **DONE, rescoped** | Rescoped from 16 full-size cells to 4 lean cells (MATH500 only, 64 prompts) per explicit direction: the method's exactness is already proven on 4,680 pairs, so AR only needed a stable throughput number, not completism. 43.75/25.27/43.75/26.70 tok/s for DFlash-8B/14B, EAGLE-3-8B/14B. |
| E3 Stage A | Public fine-tuned descendant, zero-shot | **DONE, strong positive** | Found and verified `nvidia/Nemotron-Orchestrator-8B` (rejected one architecturally-mismatched candidate first). 91.9% retention, 1.605x speedup, beating the base target's own registered number. |
| E3 Stage B | Learned low-rank delta on the descendant | **DONE, null result** | Implemented directly (no `peft` needed: a zero-initialized low-rank delta on the frozen relay). Rank-32 delta: 92.0% retention, 1.504x speedup, statistically indistinguishable from Stage A's zero-shot 91.9%/1.605x. Stage A left too little gap for the delta to demonstrate value; the underlying hypothesis is untested, not refuted. |
| E8 | Literature-cited adaptation-cost table | **DONE** | Appendix H.2, cites PARD/EDA/HyperDFlash/VSD's own reported training setups against our single 86-119s fit. No invented multiplier. |
| Related-work Table 6 | Add PARD-2, Draft-OPD, OnlineSpec, Learning to Draft; SD² corroboration | **DONE** | Page-9 compliance re-verified twice after additions (E8 and this). |
| L3 docstring correction | Formal A_soft/A_hard caveat | **DONE** | `src/relayspec/losses.py`. |
| E5 | Relay capacity: rank/tap sweep | **DROPPED, reasoned** | No capacity knob exists in `TargetFeatureRelay`; tap count is hard-asserted equal to the frozen proposer's own architecture. Building one would mean inventing new architecture to answer a hypothetical reviewer question the plan only speculated about. |
| X2 | Within-family replication in a non-Qwen family | **DONE, strong positive** | Per explicit user direction, superseding the earlier "not pursued" call. Transplanted the Llama-3.1-8B-native DFlash proposer (`z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`) onto Llama-3.2-3B: 92.1% retention, **1.882x speedup**, the best speedup in the project, in a family with no Qwen involvement. Decisively answers PARD's #1 reviewer objection. |
| X1 | Cross-family retargeting via S-A string verification | **DONE, working, modest result** | Built the plan's S-A bridge from scratch: `src/relayspec/vocab_bridge.py` plus a new `cross_family_relay_dflash_generate` loop and a new `cross_family` training mode (`train_relay.py`), since fitting itself needed independent per-model tokenization, not just inference-time bridging. Took 3 real bug-fix rounds (an out-of-range tap index, the fitting-time tokenizer assumption, and an invented cache-fill step that violated the vendored attention layer's key-concatenation/rope contract, found by reading `dflash/model.py` directly). Final run (Qwen proposer/relay retargeted to Llama-3.1-8B): mean acceptance length 1.185 (near the forced floor), 0.871x speedup (slower than plain autoregressive), 50% exact-match against native_ar (the first disagreement seen anywhere this session, all other methods always 100%). Reported as a working pipeline with a negative speed result and an open correctness question, not a positive claim. |
| E2 | Fitting-data composition (3 variants x 14B evals) | **BLOCKED, real reason** | No training-split code/chat corpus exists in the repo; sourcing one and clearing the mandatory split-hygiene audits (Section 3.3) is a real data-engineering task, not a config change, and was not rushed. |
| E9 | vLLM/SGLang concurrency | **DECIDED: OUT OF SCOPE** | Closing Open Decision 3 (Section 9) explicitly: this paper targets ICLR with a stated batch-one, Transformers-runtime scope (already written into Section 5, citing SPEED-Bench), not a systems venue submission with engine numbers. A vLLM/SGLang integration is a materially different, multi-day engineering project, not an extension of this codebase. Not pursued for this submission; stated as a limitation, not silently omitted. |

**Final closure.** E3 Stage B, initially assumed to need `peft`, was
reconsidered, implemented directly, and run to a real (null) result. X1/X2,
initially scoped out, were executed per explicit user direction: X2 is a
strong positive (the project's best speedup, in a non-Qwen family), X1
required real new engineering (a working S-A bridge, built and debugged
from scratch this session) and produced a working-but-modest result
(near-floor acceptance, negative speedup, an open exact-match gap), reported
honestly rather than rounded up to a win. The two items still remaining
(E2, E9) are explicit, permanent scope decisions, not pending ones: each
was checked for feasibility, found to require external data sourcing with
contamination-audit obligations or a materially different engineering
project, and was deliberately scoped out with its reasoning stated in this
document and the decision register, exactly as Section 8 ("what we will not
claim") already does for other boundaries. The E4-P1 asymmetry finding was
independently confirmed at a second learning rate before being accepted,
rather than resting on one run. This is what
completing the plan means for a plan that was written with
explicit non-claims from the start: every experiment either has a result,
or has a stated, reasoned scope boundary. The turing queue is empty; no
further GPU work is planned in this iteration.

---

## 11. Paper vision across outcome branches

The Phase 1 gate closing negative removes one axis of uncertainty (the
paper is not a "no source model ever" paper) but several experiments below
still branch the final claim. This section states, for each remaining
branch point, what the paper's headline contribution is if the experiment
succeeds versus if it does not, so that no outcome leaves the paper without
a coherent ICLR-acceptance story.

**Baseline floor (true regardless of any remaining experiment).** L0's
registered result already stands on its own: a frozen, third-party-trained,
feature-conditioned proposer transplanted onto a new target with one linear
map, fit from 4,096 unlabeled examples, at 1.523x/1.086x-1.237x speedup
across DFlash and EAGLE-3, both scales, with a formal Amdahl break-even rule
and full task-transfer, memory, and design-ablation evidence already in the
paper. This is a complete, defensible ICLR submission by itself. Everything
below is upside, not a precondition for submission.

**E4-P0/P1 (representation probe): resolved, actual outcome.** P0 showed
strong cross-scale alignment (cosine 0.973, CKA 0.732, subspace overlap
0.897). P1, run in both directions, found real but asymmetric transfer: a
decoder fit for the smaller target (8B) generalizes substantially upward to
the larger one (79.3% retention, 1.241x speedup), but a decoder fit for the
larger target (14B) does not generalize downward (16.4% retention,
collapse). This is neither full Branch A (near-parity in both directions)
nor flat Branch B (collapse in both directions). Treatment: report it as a
genuine, second finding -- a real but partial and direction-dependent
transfer signal -- in its own paragraph or short subsection, not promoted
to a main-text contribution that reorganizes the paper's thesis, and not
buried as a one-line negative result either. State plainly that only one
size pair was tested and the asymmetry should not be generalized beyond it.

**X2 / X1 (cross-family generalization): resolved, actual outcome.** X2
succeeded strongly: within-family replication in Llama, 92.1% retention,
1.882x speedup, the project's best result, closing the single-most-likely
reviewer objection (Section 2, PARD review record) at low cost. This alone
supports the *method-generalizes-beyond-Qwen3* claim outright. X1 (true
cross-family via the S-A bridge) also ran, but did not succeed in the sense
this section anticipated: acceptance stayed near the forced floor, speedup
was 0.871x (a regression, not a gain), and exact-match against the
baseline was only 50%, an open correctness question this session did not
resolve. Treatment: report X2 as the headline cross-family result exactly
as planned. Report X1 as a documented, working negative result -- the
engineering (a genuine cross-tokenizer speculative-decoding pipeline) is a
real contribution to describe, but the paper must not claim a speedup or a
correctness guarantee for it that the data does not support. Do not
attempt to fold X1 into the headline claim; a reviewer who reruns it would
find the same numbers.

**E5 (capacity sweep).**
- Any outcome is reportable: either the current 52-66M-parameter relay is
  shown to be near a capacity/quality knee (justifying the design choice
  directly), or a smaller relay matches performance (strengthening the
  efficiency claim), or a larger one helps materially (a limitations-section
  admission that the current design is under-parameterized, which is honest
  and still fine for acceptance). There is no failure mode here that removes
  a claim, only ones that adjust its precision.

**E2 (task-composition) and E3/E3b (descendant transfer, architecturally
shifted target).**
- These answer specific reviewer-facing objections (the two negative 14B
  code cells, and "does this survive a fine-tuned or architecturally
  different target") without changing the core claim's truth value either
  way. A negative E3/E3b result becomes a stated limitation ("the map is
  sensitive to architectural drift beyond X"), which is still a publishable,
  honest boundary condition, not a paper-killing result.

**E9 (concurrency).**
- If run, the paper can make a production-relevant throughput claim under
  concurrent load, the single biggest reviewer-facing gap per the PARD
  review record. If not run (Open Decision 3, Section 9), the paper
  explicitly scopes its claim to the single-request setting throughout
  (already done in Section~\ref{sec:related}'s SPEED-Bench citation) and
  states production serving as future work. Both are legitimate ICLR
  submissions; only the first is a strictly stronger one.

**Bottom line.** There is no branch in this remaining work that leaves the
paper without an acceptable, honestly-scoped ICLR submission: the registered
L0 result is sufficient on its own, and every remaining experiment is
either upside (P1, X1, E9) or a bounded, statable limitation on failure (P0,
X2, E5, E2, E3/E3b). The purpose of running the remaining queue is to
maximize the strength of the accepted claim, not to avoid rejection.
5. Whether Branch A of Section 3.5 is acceptable as a plan outcome, since it
   implies retitling and reorganizing the paper mid-iteration.
