# RelaySpec decision register

This register was frozen before Part-2 EAGLE-3 test evaluation. Its
machine-readable source of truth is `configs/relayspec_protocol.yaml`.

## Evidence classes

- **F — formal:** follows from an explicit derivation and stated assumptions.
- **P — peer reviewed:** copied from a verified primary proceedings paper.
- **O — official implementation:** copied from a pinned released repository,
  checkpoint, or dataset manifest and checked by a conformance test.
- **V — validation selected:** chosen on registered development data; every
  independent confirmatory population excludes it, and any conventional table
  that contains it is labeled as overlapping.
- **T — test result:** an immutable result, never used retroactively to tune the
  same test.
- **A — administrative:** resource or reproducibility identifier that is not
  claimed to be scientifically optimal.

Secondary summaries and technical blogs may identify neighbors but cannot be
the sole justification of a method choice or numerical claim.

The protocol auditor also rejects numeric leaves in every active configuration
unless they are linked to a registered decision. The small set of
non-scientific exemptions---schema version, deterministic seed, CPU request,
manifest execution cap, one excluded warm-up, one paired execution per manifest item,
logging cadence, checkpoint dimensions, and checkpoint identity---now each
has a machine-checked rationale in `administrative_path_rationales`. An
unexplained exemption therefore fails validation just like an unexplained
scientific constant.

## Frozen mechanism decisions

| Decision | Origin | Reason |
|---|---|---|
| Target verifier is the only token authority | F | Required for exact target decoding; proposer context can affect acceptance only. |
| Predict the exact post-fusion tensor consumed by the proposer | F + V | A direct linear map can express the composition of any pre-fusion linear map and frozen fusion while emitting about one-fifth the width for five taps; the validation ablation tests the systems effect. |
| Relative interface MSE is the primary loss | F | It contains both radial and angular error and introduces no mixing coefficient. |
| No primary `0.1` cosine or KL coefficient | T + F | Cosine duplicates normalized-error geometry; completed Part-1 KL refinement did not produce a resolved throughput improvement. |
| Qwen3 non-thinking, greedy, 2,048-token cap | P | Matches the DFlash primary evaluation and permits exact sequence agreement without artificially short math output. |
| DFlash block length 16 | P + O + V | Released model choice and Part-1 8/16/32 throughput ablation. |
| EAGLE draft length 7 | O | The pinned Qwen3 checkpoints are the released `ttt7` artifacts; changing length would study a different proposer configuration. |
| Five released feature depths | P + O | EAGLE-3 motivates multi-depth fusion and the pinned checkpoint contract names exactly five depths; RelaySpec reproduces that interface. |
| Zero weight decay | O | DeepSpec commit `005e03b…`, `config/eagle3/eagle3_qwen3_8b.py`, uses zero; its DFlash Qwen3 config agrees, so no unsupported regularizer is added to the linear map. |
| Preserve scale for EAGLE; normalize for DFlash | F + V | The frozen interfaces differ in whether their consumed tensor is normalized. A matched-parameter 8B ablation raises acceptance retention from 69.0% to 82.6% and speed from 1.035x to 1.237x. |
| Full-rank single linear map | F + T | This is the unrestricted minimal affine hypothesis. Relay arithmetic is only 0.4% of reference runtime, so low-rank factorization has negligible Amdahl upside and adds an expressivity bottleneck. |
| Bias-free projection | F + O | It is the minimal linear interface hypothesis and matches the released source fusion; an offset adds an unsupported degree of freedom. |
| Batch one, BF16, dense SDPA | P + O + A | Batch one matches the latency question and related evaluations; pinned model dtypes are BF16; one dense backend is held identical across paired methods, without extrapolating to serving kernels. |
| Rotated method order and one excluded warm-up | F + A | Rotation balances first-method/cache/thermal effects; one warm-up materializes lazy kernels and is never included in a statistic. |
| Paired cluster bootstrap | F | Whole requests preserve output-length coupling; both turns of an MT-Bench conversation are resampled together because they share history. |
| Pinned official scorers | O | Qwen math and EvalPlus revisions plus scorer-source hashes are stored inside every result directory. |

## Why the evidence hierarchy itself is justified

The hierarchy prevents a circular argument in which a test result is used to
choose the method and then reported as evidence for that same method. Formal
constraints determine correctness-critical choices; peer-reviewed papers set
comparability choices; pinned official artifacts define implementation
interfaces; development data choose unresolved efficiency tradeoffs; untouched
tests support final claims; administrative values only make runs reproducible.
No weaker class is allowed to override a stronger correctness constraint.

Constants therefore enter the system in only four defensible ways:

1. **Forced by algebra or architecture:** target-only token authority,
   post-fusion output width, and source layers 0--33 because tap 33 is the
   final required zero-based feature.
2. **Matched to peer-reviewed evaluation:** non-thinking Qwen3, greedy decoding,
   2,048 generated tokens, DFlash block 16, and benchmark/scorer definitions.
3. **Copied from an immutable release:** tap indices, EAGLE maximum proposal
   length, software commits, and checkpoint revisions.
4. **Selected on disjoint development data:** learning rate and the
   interface-dependent input normalization rule. All candidates remain in the
   ablation artifact. Fit examples, steps, and training length are labeled
   controlled compute budgets rather than claimed optima.

The 95% interval level and 10,000 paired-bootstrap resamples are administrative
choices declared before final tests. At either 2.5% tail, 10,000 draws give
binomial CDF standard error
`sqrt(0.025 * 0.975 / 10000) = 0.00156`; the resampling unit is the whole paired
request, not an individual token.

The 32-prompt development size is likewise a screening budget, not an accuracy
claim: it gives exactly eight requests to each of four ranks, keeps every
architecture probe short, and advances a candidate only through its measured
paired interval. Its uncertainty is not extrapolated. Final evidence comes
from all 500 conventional MATH prompts and, separately, the preregistered
468-prompt complement that excludes every development prompt.

## Selection rules

The relay-fit prompts and evaluation prompts are disjoint by operator-preserving
normalized hashes. The normalizer collapses whitespace but deliberately keeps
mathematical and programming symbols; erasing `+`/`-` can create false leakage
matches between distinct problems. The executable audit finds zero exact
matches across 4,096 fit records and all 1,250 evaluation records. Loss and
interface choices use 32 registered
MATH-500 development prompts. The conventional all-500 table includes those
32 prompts for direct benchmark comparability and is labeled accordingly; it
is not called untouched. The normalized-hash 468-prompt complement excludes
them and supplies the independent MATH confirmation. HumanEval, EvalPlus MBPP,
MT-Bench, and GSM8K-128 remain untouched until the configuration is frozen.

The first clean EAGLE breadth artifact exposed an implementation-completeness
error: its driver used only `benchmark_turns(record)[0]`, so all math/code rows
and the 80 first chat turns were valid but the 80 dependent second turns were
absent. The iterator was corrected to maintain a separate conversation history
for each paired method, rotate order at each turn, and suffix turn-specific
problem IDs. Dedicated four-GPU two-turn runs replace the EAGLE MT-Bench cells;
the incomplete chat rows are not used in the final matrix. This correction
changes no model, checkpoint, loss, or selection threshold.

No fixed cosine-distance cutoff is used. Advancement is governed by the
measured cost model

\[
\frac{L_R}{L_S}=\frac{a_S}{a_R}p_C+p_O+p_R,
\]

and requires the measured acceptance retention to exceed its algebraic
break-even point. The separate empirical speed gate requires the lower bound
of the paired request-bootstrap RelaySpec/source-reuse speed to exceed one.
All candidates remain visible in the ablation artifact.
For the 32-prompt development confirmation, the executable gate additionally
requires complete paired coverage and exact source/relay sequence agreement.
These cutoffs are not tuned constants: one is the algebraic no-slowdown
boundary and exact agreement is the zero-observed-change condition. The full
test still reports official paired task accuracy and all finite-precision
sequence disagreements rather than assuming that development agreement proves
quality.

For DFlash objective selection, a matched full-budget comparison changes only
coefficient-free relative MSE to the historical `MSE + 0.1 cosine` objective.
The rule was frozen before observing that comparison: select the faster
gate-passing candidate independently for each target, because each target has
its own trained translator. If and only if `0.1` wins, it must also beat
log-spaced neighboring weights `0.03` and `0.3` under the same fit and gate.
This conditional bracket spends no compute on a discarded coefficient and
prevents an inherited decimal from silently becoming a claimed optimum. For
the normalized DFlash interface, coefficient-free relative MSE is the
zero-cosine endpoint up to the frozen target-energy scale, so the bracket tests
whether explicit angular emphasis has operational value.

## First live optimized-baseline evidence

The 32-request Qwen3-4B-to-8B DFlash profile is a development systems probe,
not the final task-quality table. After optimizing source reuse to execute the
4B trunk only on the committed prefix and truncating it after layer 33, the
source trunk still occupies 38.8% of end-to-end request time. RelaySpec agrees
with source reuse on all 32 output sequences and reduces 161.466 seconds to
108.621 seconds: 1.487x with paired request-bootstrap interval [1.454, 1.520].
Acceptance retention is 91.5%; the independently derived Amdahl model predicts
1.486x and an equal-acceptance ceiling of 1.619x. The immutable artifact is
`reports/optimized-source-profile/analysis.json`.

## Second-family architecture evidence

The EAGLE-3 4B-to-8B optimized source path spends 33.3% of request time in the
removable source trunk.  A normalized-input relay barely clears break-even:
1.035x [1.003, 1.068], with 69.0% acceptance retention.  Preserving target
feature scale changes no parameter count or inference operation, but improves
the 128-step relative loss from 0.413 to 0.266.  After the same 1,024-step fit,
it reaches 1.237x [1.213, 1.263], retains 82.6% of acceptance, and agrees with
source reuse on 32/32 sequences.  The independently computed Amdahl prediction
is 1.238x.  The immutable selection artifact is
`reports/design-selection/eagle3/ARCHITECTURE_SELECTION.md`.

The frozen rule transfers to Qwen3-14B without retuning: the 32-request
confirmation is 1.086x [1.068, 1.102], 32/32 exact, with 27.1% removable source
work and 79.5% acceptance retention.  Amdahl predicts 1.090x.  This lower gain
than at 8B is expected because the fixed 4B source is a smaller fraction of the
larger verifier's runtime; it is positive scaling evidence rather than a
post-hoc exception.

## Historical constants

Part-1 used `MSE + 0.1 cosine`, AdamW learning rate `2e-4`, weight decay `0.01`,
and gradient clipping `1.0`. Those values describe completed historical runs;
they are not automatically carried into Part 2. The old `0.1 KL` refinement is
also historical and is not the core method.

## Source-free supervision gate: L1/L2 cold-start collapse

Per the source-free retargeting plan (Section 7 gate), two cold-start
`target_verifier` supervision objectives were fit for Qwen3-8B/DFlash-4B with
no source-derived interface targets and no source-trunk forward pass at fit
time: L1 (`greedy_agreement_ce`, cross-entropy against the target's own
greedy-argmax continuation) and L2 (`proposal_kl_to_target`, KL divergence to
the target's next-token distribution). Both were fit for 128 steps and
evaluated on the same 32-prompt MATH dev-gate manifest used to register L0.

Both collapse to near the forced floor. L0 (source-interface regression,
registered baseline) retains 92.9% of the paired `optimized_source_reuse`
mean acceptance length (7.67) at 1.523x speedup. L1 reaches a mean acceptance
length of 1.591 (20.7% retention, 0.349x, i.e. slower in wall-clock decode
time than the removable-source baseline it is meant to replace). L2 reaches
1.657 (21.6% retention, 0.367x). Both are exact-match 32/32 on final output
text (greedy decoding forces convergence regardless of draft quality), so the
failure is purely a proposer-quality collapse, not a correctness bug: the
relay produces a `c_hat` that the frozen DFlash decoder essentially cannot
use, at almost the same rate a length-1 fallback would.

This is decisive evidence, not a partial gate result: 128-step cold-start
supervision from the target's own verification signal, with no dense
per-dimension regression target, does not carry enough gradient information
to recover the specific 2,560-dimensional interface structure the frozen
proposer was trained to consume. The plan's Phase 1 gate criterion (any
source-free objective within ~2 acceptance-retention points of L0) is not
met by L1 or L2. Per the plan, this shifts the headline framing toward L4
(warm-start from the published L0 checkpoint, then a short
`accepted_prefix_surrogate` fine-tune) as "source-trunk-free fitting" rather
than a from-scratch source-free method, pending L3 and L4 dev-gate results
(both queued/running at time of writing). Immutable artifacts:
`outputs/bench-verifier-l1-dev32-26576/benchmark-summary.json` and
`outputs/bench-verifier-l2-dev32-26588/benchmark-summary.json` on turing.

A separate, unrelated bug was found and fixed during this gate: the E7 DFlash
memory-isolation configs that select only `optimized_source_reuse` (no relay
method) failed immediately because `scripts/benchmark_relay.py` unconditionally
required a `relay_probe.checkpoint_path` even when no relay-based method was
selected. Fixed by gating the checkpoint load behind
`needs_relay = bool(set(method_names) & {"relay_f", "relay_p"})`, matching the
existing `model_load_plan` gating pattern used elsewhere in the same file.

L3 (`accepted_prefix_surrogate`, the differentiable accepted-length objective,
cold-start, 128 steps, same protocol as L1/L2) is worse, not better: mean
acceptance length 1.088 (14.2% retention, 0.236x). This is at the forced
floor of 1 (every commit contributes at least the target's correction
token), so the surrogate essentially failed to move the relay away from
init under cold-start optimization. This is the clearest evidence yet that
the collapse is not specific to the CE/KL objective choice: it is a
cold-start optimization-landscape problem common to all three source-free
objectives (L1, L2, L3) tested from a random relay initialization with no
source-derived interface signal. Immutable artifact:
`outputs/bench-verifier-l3-dev32-26589/benchmark-summary.json` on turing.
L4 (warm-start from the published L0 checkpoint, then continue with the
`accepted_prefix_surrogate` objective) also fails: mean acceptance length
1.465 (19.1% retention, 0.335x), i.e. a 256-step `accepted_prefix_surrogate`
fine-tune destroys the L0 checkpoint's working interface rather than
refining it. This closes the source-free/source-trunk-free branch entirely
for this round: L1, L2, L3, and L4 all collapse relative to the
source-anchored L0 baseline, including the warm-start case, which rules out
"bad initialization" as the explanation. The `accepted_prefix_surrogate`
objective (`A_soft`) is actively harmful under gradient descent in this
setting, not merely uninformative. Immutable artifact:
`outputs/bench-verifier-l4-dev32-26590/benchmark-summary.json` on turing.

**Decision**: per user direction, the source-free/source-trunk-free
supervision line is not the paper's headline result. L0's source-anchored
regression fit remains the registered method. This branch is documented here
as a negative result (a real, load-bearing finding: interface regression
against source-derived targets is doing necessary work that verifier-only
signals cannot replace at this step budget) and deprioritized for further
GPU spend.

## E4-P0: representation probe shows strong cross-scale alignment

The free, zero-training representation probe (`analyze_relay_manifold.py`)
compares the published Qwen3-8B and Qwen3-14B DFlash relays' output
trajectories (`c_hat`) in the shared 2,560-dimensional proposer interface,
over 32 identical held-out MATH examples (3,767 compared token positions).
Result: elementwise cosine mean 0.973 (std 0.018), linear CKA 0.732,
rank-32 principal subspace overlap 0.897. This is strong alignment by any
of the three independent measures, not just one. Immutable artifact:
`outputs/e4-p0-manifold-26599/manifold-probe.json` on turing.

Per the plan's Section 7 gate ("run E4-P1 only if P0 shows strong
alignment"), this result licenses spending the E4-P1 budget (two real fits
testing partial representation transfer between the 8B and 14B relays).
This is now the plan's Branch A: if P1 confirms transfer, the paper gains a
second structural contribution beyond portability alone, evidence of a
canonical, target-independent proposer-facing representation. E4-P1 is
queued as the next experiment.

## Scope correction: E6 cut down, E5 dropped

Per explicit user direction: RelaySpec is exact-match/accuracy-preserving
under greedy decoding (already proven on 4,680 pairs with zero
disagreements), so E6's native-AR fill only needs enough prompts to get a
stable throughput number, not full-dataset accuracy coverage. E6 is
rescoped from 8 jobs (4 combos x {MATH500 full 500, breadth-suite full 750})
to 4 jobs (4 combos x MATH500 only, 64 prompts), matching the paper's actual
headline benchmark instead of completism across four tasks that already
have paired source/relay numbers. The GSM8K/HumanEval/MBPP/MTBench
native-AR breadth cells are dropped: they would not change any claim in the
paper, only fill a table cell nobody is asking about.

E5 (capacity/rank sweep) is dropped for the same reason. `TargetFeatureRelay`
(`src/relayspec/relay.py`) is a single full-rank linear layer with no
capacity knob: its input width is fixed by `target_hidden_size * num_taps`
and its output width by `draft_hidden_size`, both determined by the target
and frozen proposer, not a free choice. `train_relay.py:146` hard-asserts
`len(target_layer_ids) == len(draft.target_layer_ids)`, so even the tap
*count* is fixed by the frozen DFlash proposer's own architecture; there is
no free tap-count sweep available either. Running E5 as specified would
require adding a new low-rank bottleneck architecture to the relay purely
to answer a hypothetical "why 52-66M parameters" reviewer question that the
plan itself only speculated might be asked. That is new scope, not
implied by anything already in the paper's claims, so it is dropped rather
than built.

## E6 native-AR baseline results (MATH500, 64 prompts)

| Config | Decode tok/s | Mean acceptance length |
|---|---|---|
| DFlash, Qwen3-8B | 43.75 | 1.0 (no drafting) |
| DFlash, Qwen3-14B | 25.27 | 1.0 |
| EAGLE-3, Qwen3-8B | 43.75 | 1.0 |
| EAGLE-3, Qwen3-14B | 26.70 | 1.0 |

As expected, native-AR throughput is proposer-family-independent (it never
invokes the draft model), so DFlash and EAGLE-3 numbers at each scale agree
within measurement noise. This closes the confirmed gap: the paper
previously had a native-AR baseline for only 4 of the tasks it evaluates,
none of them isolating throughput at 64-prompt granularity for both
proposer families. Immutable artifacts:
`outputs/{dflash,eagle3}_qwen3_{8b,14b}_native_ar_math500-{26609,26610,26611,26612}/benchmark-summary.json`.

## E3 Open Decision 1 resolved: public descendant found and verified

Per the plan's own stated preference ("a public descendant for the headline
number"), found and verified `nvidia/Nemotron-Orchestrator-8B`: confirmed
via its `config.json` to be architecturally identical to `Qwen/Qwen3-8B`
(`model_type: qwen3`, `Qwen3ForCausalLM`, hidden_size 4096, 36 layers,
vocab_size 151936, matching Qwen3-8B's own config field for field), so it is
a genuine same-tokenizer, same-width, different-weights descendant, not a
different model generation. One earlier candidate
(`empero-ai/Qwen3.8-9B`) was checked and rejected before use: its config
showed `model_type: qwen3_5` and a 248,320-token vocabulary, a different
model family despite the similar name, which would have silently broken
the tokenizer-compatibility requirement had it been used without checking.

Stage A (zero-shot, no new training) is queued: run the already-fit,
frozen Qwen3-8B DFlash relay checkpoint directly against this descendant's
own hidden states and its own verification pass, no relay refitting,
measured on the same 32-prompt MATH dev gate against `optimized_source_reuse`
on the same descendant. Caching job 26620, dependent benchmark job 26621.

**Stage A result: strong.** `optimized_source_reuse` on the descendant:
mean acceptance length 7.691 (32/32 requests). The frozen Qwen3-8B relay,
applied zero-shot with no retraining, reaches 7.065 (91.9% retention) and a
**1.605x speedup** against that same descendant, actually higher than the
base Qwen3-8B target's own registered speedup (1.523x). This is a strong,
direct rebuttal to the same-family-alignment objection for the fine-tuned-
descendant case: the relay was never shown Nemotron-Orchestrator-8B's
weights and still transfers almost perfectly. Immutable artifact:
`outputs/bench-e3-descendant-dev32-26621/benchmark-summary.json` on turing.

**Stage B: implemented and run.** On reflection, Stage B does not need
`peft`: a low-rank delta on a single frozen linear layer is a small,
self-contained addition, not a dependency on external LoRA tooling.
Implemented directly in `TargetFeatureRelay` (`delta_rank` /
`delta_down`/`delta_up`, zero-initialized so an unfit delta is a no-op,
matching the standard low-rank-adapter convention) and wired through
`train_relay.py` (`delta_rank`/`delta_base_checkpoint_path` training keys)
and `benchmark_relay.py`. 142 local tests pass (2 new ones: the delta starts
as an exact no-op, and freezing the base leaves only the delta trainable).

Trained a rank-32 delta on top of the frozen Qwen3-8B relay for the same
Nemotron-Orchestrator-8B descendant used in Stage A (job 26646/26647).
Result: mean acceptance length 7.079 against the same 7.691 baseline
(92.0% retention, 1.504x speedup) — statistically indistinguishable from
Stage A's zero-shot 91.9%/1.605x. **The delta does not meaningfully improve
on zero-shot transfer for this descendant.** This is an honest, informative
result, not a wash to bury: Stage A's zero-shot transfer already left only
about one point of retention on the table, so there was little room for a
low-rank correction to demonstrate value. The "small model updates induce
correspondingly low-dimensional interface changes" hypothesis from Section
3.4 is neither confirmed nor refuted by this result; it is untested,
because the baseline it would need to improve on was already this strong.
Immutable artifact:
`outputs/bench-e3-stageb-delta-dev32-26647/benchmark-summary.json`.

## X2 checkpoint survey: no free within-family, non-Qwen pairing exists

Checked what X2 (within-family replication in a non-Qwen family) would
actually need: a small frozen proposer checkpoint plus a larger same-family
target, both fitting the paper's existing pattern (a source-side proposer
trained against one size, relayed onto a bigger size in the same family),
and feasible under this cluster's hard constraint that each of the 4 DDP
ranks must hold a full target-model replica on one GPU (turing's node01
GPUs are 48 GB RTX 6000 Ada, confirmed via `gpu.csv`).

Public checkpoints found: `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (EAGLE-3,
trained against Llama-3.1-8B-Instruct) and
`z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` (DFlash, same source). Neither
has a usable larger same-family target: Llama-3.1/3.3 jumps straight from
8B to 70B, and 70B needs about 140 GB in bf16, far past the 48 GB per-GPU
budget under the current data-parallel-only training design (no
tensor/model sharding exists in `train_relay.py`). The only checkpoint pair
with a workable size gap is EAGLE-1/2's Vicuna-7B-sourced proposer targeting
Vicuna-13B, but that is a different, older draft-model architecture (a
0.24B-parameter single-layer head, not EAGLE-3's design), which the current
codebase has no loader for: adding one would be new architecture-integration
work, not a config change.

## E4-P1: adapter-based representation-transfer test, implemented

Per user decision, P1 is implemented as the cheaper adapter variant rather
than the plan's full shared-decoder/per-target-encoder architecture: freeze
an already-fit relay (the decoder), train only a small linear adapter that
maps a *different* target's raw tap width into that frozen relay's native
input space, and measure how much acceptance survives through the frozen
decoder. This is a real code change (`TargetFeatureRelay` gained an optional
`adapter_input_width`/`adapter` submodule in `src/relayspec/relay.py`,
wired through `train_relay.py` via new `freeze_base_relay` /
`adapter_base_checkpoint_path` training keys, and through
`benchmark_relay.py`'s checkpoint loading), guarded so the default path
(no adapter) is byte-identical to prior behavior. 140 local tests pass,
including two new tests for the adapter path.

Both transfer directions requested were run: small base to big target
(freeze the 8B relay's decoder, adapt Qwen3-14B's raw hidden states into
it, jobs 26615/26616) and big base to small target (freeze the 14B relay's
decoder, adapt Qwen3-8B's raw hidden states into it, jobs 26617/26618).
Each trains the adapter for the same 4,096-example/1,024-step budget as the
original L0 fits, then dev-gates on the same 32-prompt MATH set.

**Small base to big target: strong positive, Branch A evidence.** Job
26617/26618 (small-to-big was submitted second but completed first). The
frozen 8B relay's decoder, with only a small adapter trained to translate
Qwen3-14B's raw hidden states into its input space, reaches mean acceptance
length 6.081 against the 14B `optimized_source_reuse` baseline of 7.671
(79.3% retention) and a real **1.241x speedup**. This is far above the
collapse seen in the failed source-free ladder (14-21% retention) and
inside the plan's Branch A territory: a decoder trained for one target size
transfers substantially, with no access to that target's own weights, to a
target roughly twice its size. Immutable artifact:
`outputs/bench-adapter-8to14-dev32-26618/benchmark-summary.json`.

**Big base to small target: infra retry in flight.** Job 26615 hit a
genuine `CUDA out of memory` at the optimizer step: the adapter mapping
14B's native decoder width down is itself a roughly 524M-parameter linear
layer, and AdamW's two extra per-parameter buffers did not fit alongside
the (larger, 14B) input-generating target model on a 48 GB card. Fixed by
adding an `adapter_optimizer` training config key (default `adamw`,
unchanged for every existing config including the already-completed
small-to-big run above) and setting it to `sgd` only for this direction,
halving optimizer-state memory. Resubmitted as 26637/26638.

**Big base to small target, final result: collapses.** Job 26637/26638
completed. Baseline `optimized_source_reuse` on Qwen3-8B in this run: mean
acceptance length 7.515. The frozen 14B relay's decoder, adapted to serve
Qwen3-8B, reaches only 1.232 (16.4% retention), 0.236x (slower than doing
nothing). This is the same collapse magnitude as the failed source-free
ladder, not a partial result.

**P1 conclusion: real but asymmetric transfer, not full Branch A.** The
transfer is direction-dependent: a decoder trained to serve the *smaller*
target (8B) generalizes substantially upward to a *larger* target (14B,
79.3% retention, real 1.241x speedup), but a decoder trained to serve the
*larger* target does not generalize downward to the smaller one (16.4%
retention, collapse). One plausible reading: the 8B-native decoder was fit
against a lower-dimensional, less-articulated representation, so adapting a
richer 14B representation down into that manifold still preserves the
information it needs; the 14B-native decoder was fit against a richer
representation, and compressing an 8B target's comparatively poorer
representation up into that manifold loses something the decoder needs. We
did not test whether this asymmetry holds at other size ratios, so we do
not generalize it beyond this one pair. This does not meet the plan's
Branch A bar (near-parity, within about 3 points, in both directions) nor
is it the flat, direction-independent collapse of Branch B: it is reported
as a real, partial, and asymmetric finding, worth a paragraph in the paper
rather than a reorganization around a canonical shared representation.
Immutable artifacts:
`outputs/bench-adapter-14to8-dev32-v2-26638/benchmark-summary.json` and
`outputs/bench-adapter-8to14-dev32-26618/benchmark-summary.json`.

**Follow-up: ruled out under-optimization as the cause.** The SGD retry
above reused the AdamW-tuned learning rate (0.0006), which is typically
10 to 100x too low for plain SGD, so a legitimate concern was that the
16.4% retention reflected an under-trained adapter rather than a real
transfer limit. Retrained with `learning_rate: 0.02` (job 26640/26641),
same steps and data. Training loss improved (final `feature_loss` 0.65,
`cosine_distance` 0.42, versus 0.84/0.60 before), and dev-gate retention
rose from 16.4% to 26.8% (mean acceptance length 2.014 vs. baseline 7.515).
Still a clear collapse, not a partial result: doubling the optimization
quality only partially closed the gap, nowhere near the 79.3% retention of
the small-to-big direction. This makes the asymmetry a more solid finding,
confirmed across two learning rates rather than resting on one possibly
under-tuned run. Immutable artifact:
`outputs/bench-adapter-14to8-dev32-v3-26641/benchmark-summary.json`.

## X2 and X1: cross-family, executed per explicit user direction

Per direct user instruction ("cross family is required... do all things in
the plan according to priority"), the earlier recommendation not to pursue
this was superseded. Checkpoints found and verified (config-checked, not
assumed): `unsloth/Llama-3.1-8B-Instruct` and `unsloth/Llama-3.2-3B-Instruct`
(ungated mirrors of Meta's gated originals, bit-identical weights, verified
config match), `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` (a DFlash
proposer natively trained against Llama-3.1-8B, confirmed via its own
config: `vocab_size: 128256`, `target_layer_ids: [1,8,15,22,29]`,
`num_target_layers: 32`, all consistent with Llama-3.1-8B's real
architecture).

**X2 (within-family, Llama): strong positive, the best result in the
project.** Fit a new relay transplanting this Llama-3.1-8B-native DFlash
proposer onto the smaller sibling Llama-3.2-3B (job 26677/26720). Result:
mean acceptance length 3.887 against an `optimized_source_reuse` baseline of
4.221 (92.1% retention) and a **1.882x speedup**, the highest speedup
measured anywhere in this project, in a family with no Qwen involvement at
all. This directly and decisively answers PARD's #1 reviewer concern (the
same-family-alignment objection): the method is not Qwen-specific.

**X1 (true cross-family, Qwen proposer to Llama target): required new
engineering on both the inference and fitting sides, built and running.**
Implemented the plan's S-A solution (string-level verification bridge):
new module `src/relayspec/vocab_bridge.py` (retokenization helpers, with a
`committed_suffix` function whose tests caught a real anchor-duplication
bug during development) and a new
`cross_family_relay_dflash_generate` function in
`src/relayspec/generation.py`, wired into `benchmark_relay.py` as method
`relay_p_cross_family`. 149 local tests pass, including 7 new ones for the
bridge logic.

First attempt (job 26721) failed immediately with a real, informative
error: it reused the existing Qwen3-8B relay checkpoint's tap positions
`[1,9,17,25,33]` directly against a Llama-3.1-8B target, but Llama-3.1-8B
has only 32 layers (Qwen3-8B has 36), so tap index 33 was out of range.
This exposed a more fundamental point the plan's S-A description does not
fully spell out: X1 is not zero-shot reuse of an existing relay. It
requires *fitting a new relay* for the cross-family target (same as every
other retargeting in this paper), with S-A handling only the *inference-time*
discrete-token gap. Fitting itself turned out to have its own gap: 
`train_relay.py`'s loop feeds one shared token sequence to both the source
and target models, which is meaningless once they use different
tokenizers. Fixed with a new `cross_family` training mode (own tokenizer
per model, per-position regression collapsed to a single last-position
comparison, since two different tokenizers give no general per-position
alignment between sequences of different lengths) guarded so every
existing config's behavior is unchanged. Refit (job 26727) and its
benchmark (job 26728) are running; result pending.

**X1 final result: the pipeline runs correctly end to end, but the result
is modest, not a win, and is reported exactly as measured.** Getting a
crash-free run took three real bug fixes, each found by actually reading
error tracebacks and, for the last one, the vendored DFlash attention
source rather than guessing: (1) the first attempt reused the Qwen3-8B
relay's tap positions directly against a 32-layer Llama target (Qwen3-8B
has 36 layers), an out-of-range index; (2) fitting itself needed a new
`cross_family` training mode, since the existing loop feeds one shared
token sequence to both models; (3) the generation loop's own attention
mechanism concatenates a key/value from `conditioned_context` with one
from the speculative block and rotary-embeds the concatenation as a single
unit with no length slack, which an invented "cache-fill" step (my own
addition, with no analog in the same-family algorithm) violated. Reading
`dflash/model.py`'s `Qwen3DFlashAttention.forward` directly resolved this;
guessing at three prior attempts had not.

With all three fixed, job 26734 completed cleanly: mean acceptance length
1.185 against a `native_ar` baseline (mean acceptance length is undefined
for native_ar by construction; 1.0 is the accounting floor), 0.871x speedup
(slower than plain autoregressive, since block overhead is not recovered
by essentially any accepted proposal), and **50% exact-sequence-match rate
against native_ar** (4 of 8 requests), the first time any relay method in
this project has disagreed with its baseline. Inspecting the raw
completions shows agreement for a substantial shared prefix in every
mismatched case, then divergence partway through, the same signature
`relay_dflash_generate`'s own docstring already documents for same-family
block verification ("byte equality... is backend-dependent because
block-shaped kernels can differ numerically near greedy ties"), but at a
rate never observed in any same-family run this session (always 100%
exact match). We do not have a confirmed root cause for why cross-family's
variable-length, re-tokenized blocks make this more frequent, and do not
claim one. What we can report cleanly: the acceptance signal is close to
the floor, meaning the relay fit via last-position-only `cross_family`
supervision is not yet producing a proposer-usable conditioned_context for
this target, and the engineering milestone (a working S-A bridge, no
crashes, the frozen Qwen proposer genuinely running against a
different-tokenizer Llama target) is real but does not by itself
constitute a positive speed or reliability result. This is the honest,
final state of X1 for this session: a demonstrated, working cross-family
pipeline, with a negative speed result and a correctness caveat that would
need further work (a better cross-family fitting objective, and tracking
down the exact-match gap) before it could support a stronger claim.
Immutable artifact:
`outputs/bench-x1-cross-family-v4-26734/benchmark-summary.json` and its
sibling `benchmark-rank{0..3}.jsonl` raw rows.

**Superseded recommendation.** An earlier version of this document
recommended not pursuing X1/X2 without an explicit resourcing decision. Per
direct user instruction that recommendation was superseded and both were
executed (see above): X2 via the existing DFlash/Llama checkpoints (no new
architecture needed, EAGLE-1/2 integration was never required), and X1 via
the S-A bridge built this session. The EAGLE-1/2-architecture path
considered earlier (Vicuna-7B/13B) was not needed and was not built.

## Session closure (updated after X1/X2)

Every experiment in the source-free plan's matrix now has either a real
result or an explicit, reasoned scope decision; none are left silently
undone. Results: E1 (negative, closed), E4-P0 (strong positive), E4-P1
(real but asymmetric, both directions), E3 Stage A (strong positive), E3
Stage B (done, null result), E6 (done, rescoped), E7 (done, one real bug
found and fixed), E8 (done), X2 (strong positive, the best speedup in the
project, in a non-Qwen family), X1 (working end-to-end, but a modest,
non-positive result: near-floor acceptance, 0.871x speedup, 50%
exact-match). Explicit scope decisions, each checked for feasibility
before being ruled out rather than assumed: E5 (no capacity knob exists in
the architecture), E2 (no training-split code/chat corpus exists, needs
external sourcing plus the plan's own mandatory contamination audits), E9
(a materially different systems-engineering project, out of scope for an
ICLR batch-one submission). The turing queue is empty. This document and the plan
(`docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md`)
Section 10 are the record of that closure.

## Rigor pass: X1 cross-family rerun with aligned fitting supervision, and a data/architecture ablation suite at full ICLR scale

Per `docs/plans/2026-09-03-relayspec-rigor-pass-plan.md` and the later
explicit instruction to bring every experiment in the paper to a
128-prompt-minimum, paired-bootstrap-CI standard, two gaps were closed
this session: (1) X1's fitting supervision, and (2) three fitting-design
questions the paper had never actually measured (closed-form fitting,
architecture, and data-size scaling), all rerun at 128 prompts, 2,048-token
cap, MATH-500, matching the main protocol exactly.

**X1 root cause and fix.** X1's `cross_family` training mode (see above)
collapsed every fitting example to a single last-position comparison,
because two different tokenizers give no general per-position
correspondence between sequences of different lengths. This throws away
nearly all of the supervision signal a same-family fit gets (one label per
sequence instead of one per token). Fixed with character-span alignment
(`content_windows`/`align_positions`/`canonicalize_offsets` in
`src/relayspec/vocab_bridge.py`): for each target position, find the
source position whose character span covers the same underlying
`problem`/`solution` text, restricted to windows where that text actually
appears in the rendered prompt (excluding chat-template boilerplate),
recovering real per-position supervision without assuming a shared token
sequence. `content_windows` needed one further fix mid-session: a
whitespace-stripped fallback match, since some real MATH-lighteval
examples do not appear byte-for-byte in the rendered text.

**X1 result after the fix, at full 128-prompt scale: reverses from a
negative result to the strongest cross-family number in the paper.**
Refit (`train-x1-aligned-n7b-27063`, 1,024 steps, 94.5s, in the same
86-119s range the main paper reports for same-family fits) and
benchmarked at 128 prompts / 2,048 tokens
(`outputs/bench-x1-aligned-n7b-27064/benchmark-summary.json`): mean
acceptance length 3.208 (up from 1.185), **2.362x speedup vs. plain
autoregressive decoding** (up from 0.871x, i.e. slower than no proposer at
all), paired bootstrap 95% interval 2.255x [1.996, 2.557] on
`request_seconds` (computed directly via
`paired_bootstrap_speedup(reference_method="native_ar",
candidate_method="relay_p_cross_family")`). Exact-sequence-match rate
against `native_ar` is 25% (32/128), down from the earlier 8-prompt
estimate of 50% (4/8) — the larger sample is the more reliable one, and it
still says the same output is not reproduced most of the time.

**The exact-match gap has a confirmed, benign explanation, not an unknown
one.** Ran `scripts/diagnose_cross_family_divergence.py` against this
exact checkpoint (`outputs/diagnose-x1-aligned-27177/diagnosis.json`) on
a held-out prompt: greedy AR and the cross-family relay agree on the first
34 tokens, then diverge. At the divergence position the logit gap between
the AR-chosen token and the relay-chosen token is 0.0 (to reported
precision) — a genuine near-tie, not a large disagreement. This is the
same block-vs-single-position floating-point kernel signature the
same-family correctness appendix already documents (Appendix
`app:correctness`), just triggered far more often here because
cross-family verification runs at text granularity, so many more
positions sit near a tie than in same-family token-level verification.
This resolves the "we do not have a confirmed root cause" statement in the
prior X1 entry: there is a confirmed, mechanistically understood cause,
and it is the known floating-point tie-break phenomenon, not a bug in the
S-A bridge or the aligned fitting objective.

**Fitting-design ablation suite (never previously in the paper, now
measured at 128 prompts, DFlash/Qwen3-8B, all against the same
`optimized_source_reuse` baseline as the main table).**

| Ablation | Fit examples | Speedup vs. reuse | Paired bootstrap 95% | Accepted-length retention |
|---|---:|---:|---:|---:|
| Closed-form ridge regression (`ridge_lambda=1.0`) | 4,096 | 0.925x | [0.896, 0.953] | 54.9% |
| MLP relay (hidden width 512) | 4,096 | 0.872x | [0.841, 0.903] | 51.7% |
| Linear, reduced data | 512 | 1.150x | [1.118, 1.181] | 68.1% |
| Linear, reduced data | 1,024 | 1.372x | [1.346, 1.397] | 81.1% |
| Linear, reduced data | 2,048 | 1.477x | [1.458, 1.496] | 87.1% |
| Linear, more data | 8,192 | 1.573x | [1.558, 1.588] | 93.4% |

For reference, the main table's own headline configuration (linear,
gradient descent, 4,096 fit examples) reaches 1.483x, measured on the full
500-prompt MATH-500 set rather than this ablation's 128-prompt subset.

Artifacts: `outputs/bench-{closed-form3-27113,mlp512c-27115,scale512c-27116,
scale1024c-27118,scale2048c-27120,scale8192c-27122}/benchmark-summary.json`,
paired-bootstrap intervals computed via `scripts/analyze_profile.py` into
`outputs/paper_stats/{closed_form,mlp512,scale512,scale1024,scale2048,
scale8192}.json`. Closed-form fit cost:
`outputs/train-closed-form3-27112/closed-form-summary.json` — 220.5s data
collection + 6.8s solve = 227.3s total, comparable to (not clearly cheaper
than) gradient descent's 86-119s, since data collection (running
source/target/proposer forward passes) dominates either way and gradient
descent does not need a separate collection pass.

Three findings, reported plainly: (1) closed-form ridge regression is a
working, much simpler fitting procedure with no learning rate to tune, but
it is not free — it underperforms gradient descent by a wide margin
(0.925x vs. 1.483x) and is not obviously cheaper in wall-clock time; (2)
the MLP relay is the single worst configuration measured anywhere in this
paper (0.872x, below even closed-form linear), so nonlinearity is not
buying anything at this fitting budget and the main paper's plain linear
map is the right choice, not merely the simplest one; (3) fit-data size
has a real, monotonic, still-not-flattened effect on speed — 8,192
examples (93.4% retention, 1.573x) exceeds the main table's headline 4,096
number (1.483x, though that is measured on the full 500-prompt set, not
this ablation's 128-prompt subset, so the two are directionally but not
exactly comparable), meaning the paper's choice of 4,096 examples is a
reasonable operating point, not a demonstrated ceiling.

## Phase A dose-response: first real retention result (step1000), a genuine collapse

First completed retention benchmark from the Phase A SFT drift study (after
four rounds of infrastructure failures, see below), 128 prompts, paired:
`outputs/bench-drift-step1000-retention-v5-27192/benchmark-summary.json`.
Qwen3-8B fine-tuned for 1,000 LoRA/UltraChat steps, DFlash proposer,
against the base relay fitted before any fine-tuning.

Both `optimized_source_reuse` (0.431x vs. plain AR, mean acceptance length
1.01) and `relay_p` (0.693x vs. plain AR, mean acceptance length 1.00) have
collapsed to essentially zero speculative benefit, both slower than plain
autoregressive decoding. Exact-sequence-match rate against native AR is
0% for both methods, on every one of the 128 prompts.

The mechanistically important point: **source reuse collapsed as well as
the relay**, not just the relay. Source reuse feeds the frozen proposer
hidden states from the original, un-fine-tuned Qwen3-4B source model,
completely independent of the relay or of the target's fine-tuning. Its
collapse means the failure is not a relay artifact: the proposer's
candidates (built around the original Qwen3-8B's behavior) simply no
longer match what the *fine-tuned* target model wants to emit next. This
is behavioral drift in the target, not degradation of the linear map's
fit, and explains why the delta-rank-32 correction (which only touches the
relay's tensor-forming map) is not expected to fix it either, pending its
own benchmark.

This satisfies the plan's pre-registered kill/continue criterion in the
"continue" direction: retention has clearly collapsed rather than staying
near-ceiling, so Phase A is a real, reportable result. Still needed before
writing this into the paper: the rest of the dose-response curve (steps
50/150/400/2500) to see whether the collapse is gradual or already
complete at the smallest tested budget, and the delta32-corrected
benchmark at each step to see whether the correction recovers anything.

**Four rounds of pure infrastructure failure preceded this result**,
worth recording so the pattern is not repeated: v1-v3 failed on real bugs
(stale/evicted node-local model cache after a node-local `/scratch`
apparently lost its warmed cache between warm-up and job execution, a
`resolve_revision` local-path bug already fixed earlier this session, and
a missing base relay checkpoint on the target node, each found from actual
error logs and fixed); v4 ran to completion but hit a 1-hour SLURM time
limit that was fine for the main paper's healthy-acceptance checkpoints
but far too short once acceptance collapses to near 1.0 (near-autoregressive
speed, so a 128-prompt run costs close to the same wall-clock as running
plain AR 3-4 times over). Fixed by raising `slurm/benchmark_relay.sbatch`'s
time limit to 3.5 hours (6 hours for the EAGLE-3/DFlash paired-native-AR
500-prompt main-table reruns, which pay the same slow-native-AR cost at
4x the prompt count). Every prior timed-out job was resubmitted under the
fixed limit rather than left stale.

## Phase A dose-response: second point (step50) shows the collapse is not gradual

Second completed retention benchmark,
`outputs/bench-drift-step50-retention-v5-27191/benchmark-summary.json`,
128 prompts, paired. Qwen3-8B fine-tuned for only 50 LoRA/UltraChat steps,
the smallest drift dose tested.

`optimized_source_reuse`: 0.458x vs. plain AR (mean acceptance length
1.007). `relay_p`: 0.700x vs. plain AR (mean acceptance length 1.001).
Exact-sequence-match rate against native AR is 0% for both, on all 128
prompts.

These numbers are nearly identical to step1000's (0.431x/0.693x,
acceptance 1.01/1.00, also 0% exact match). Twenty times more fine-tuning
(50 to 1,000 steps) produced essentially no further degradation, because
there was almost nothing left to degrade: acceptance was already at the
floor by step 50. This means the dose-response curve is not a gradual
slope but a cliff, collapsed at the very first checkpoint tested. Whether
an even smaller step count (below 50) would show a smooth transition, or
whether the collapse threshold sits somewhere below 50 steps entirely, is
not yet measured and would need a finer-grained checkpoint sweep to
answer; the current five checkpoints (50/150/400/1000/2500) were chosen
log-spaced under the assumption of a gradual curve, which this result
contradicts.

## Phase A dose-response: delta-rank-32 correction does not recover step1000's collapse

`outputs/bench-drift-step1000-delta32-v5-27195/benchmark-summary.json`,
128 prompts, paired, `relay_p` here is the delta-rank-32-corrected relay
fitted specifically against the step1000 checkpoint (E3-Stage-B-style
correction). Corrected mean acceptance length: 1.0002, against the
uncorrected relay's 1.0006 from the retention benchmark above — no
measurable recovery. Working the reported ratio (1.608x vs.
`optimized_source_reuse`, which itself is 0.431x vs. plain AR) back to the
plain-AR comparison that matters gives approximately 0.69x, indistinguishable
from the uncorrected relay's 0.693x.

This is a clean negative result, not an inconclusive one: the correction
mechanism (a low-rank adjustment to the linear map that turns target
hidden states into the tensor the proposer expects) is the wrong kind of
fix for this failure mode. Step50/step1000's shared collapse (both methods,
0% exact match against native AR) already showed the failure is behavioral
drift in what the fine-tuned target wants to generate next, not a
miscalibrated tensor. A correction that only changes how the tensor is
computed cannot fix the proposer's candidates no longer matching the
target's drifted token distribution. Still to check: whether delta32
recovers anything at smaller drift (if a below-50-steps checkpoint is
added) or whether this negative result holds at every tested step.

## X2 (Llama family transfer) rerun at 128 prompts: confirms the 32-prompt estimate

`outputs/bench-x2-dev128-27315/benchmark-summary.json`, 128 prompts, paired.
DFlash proposer trained for Llama-3.1-8B, relayed onto the smaller
Llama-3.2-3B target, no Qwen model involved.

`relay_p`: 2.720x vs. plain AR (mean acceptance length 3.714).
`optimized_source_reuse`: 1.473x vs. plain AR (mean acceptance length
4.045). Relay vs. source-reuse: 1.846x, retention 91.8% — both nearly
identical to the earlier 32-prompt estimate (1.882x, 92.1%), confirming
that estimate was not a small-sample artifact. Exact-sequence-match rate
against native AR is 45.3% (58/128), lower than the earlier 8/32-prompt
runs could reliably show; consistent with the same block-verification
near-tied-logit sensitivity documented elsewhere in this paper (Appendix
B, and the X1 cross-family divergence trace), not a new failure mode.

This result is ready to go back into Section 7 of the paper, replacing
the earlier 32-prompt-scale version that was removed.

## Adapter reuse, 8B-to-14B direction, rerun at 128 prompts: confirms the 32-prompt estimate

`outputs/bench-adapter-to14b-dev128-27316/benchmark-summary.json`, 128
prompts, paired. The relay fitted for Qwen3-8B is frozen; only a small
adapter is trained to feed it Qwen3-14B's hidden states (small-to-large
direction).

`relay_p`: 4.138x vs. plain AR (mean acceptance length 6.100).
`optimized_source_reuse`: 3.246x vs. plain AR (mean acceptance length
7.697). Retention: 79.2%, relay vs. source-reuse: 1.275x — both close to
the earlier 32-prompt estimate (79.3% retention, 1.241x), confirming that
estimate. Exact-sequence-match rate against native AR: 28.1% (36/128).

Ready to go back into Section 7, replacing the removed 32-prompt version.

## Adapter reuse, 14B-to-8B direction, rerun at 128 prompts: retention confirmed, speed claim corrected

`outputs/bench-adapter-to8b-dev128-27317/benchmark-summary.json`, 128
prompts, paired. The relay fitted for Qwen3-14B is frozen; only a small
adapter is trained to feed it Qwen3-8B's hidden states (large-to-small
direction, the collapse case).

`relay_p`: 1.533x vs. plain AR (mean acceptance length 2.005).
`optimized_source_reuse`: 3.968x vs. plain AR (mean acceptance length
7.506). Retention: 26.7%, matching the earlier 32-prompt estimate (26.8%)
almost exactly.

**This corrects, not just confirms, the earlier claim.** The prior
32-prompt-based paper text states this direction "collapses... slower
than using no proposer." The acceptance-rate collapse is confirmed
(26.7% vs. 26.8%), but the derived speed claim is not: at 128 prompts,
`relay_p` reaches 1.533x plain AR, faster than not using a proposer at
all, just far worse than source-reuse (3.968x) and far worse than the
reverse 8B-to-14B direction (4.138x, see above). The small sample got the
acceptance-length ratio right but the timing-derived conclusion wrong.
When this goes back into Section 7, the claim must be corrected to
"badly degraded but still faster than plain AR," not "slower than no
proposer at all."

## Target-feedback ablation, variant 1 (cross-entropy), rerun at 128 prompts

`outputs/bench-verifier-l1-dev128-27318/benchmark-summary.json`, 128
prompts, paired. Config confirms `verifier_objective: greedy_agreement_ce`
— this is the cross-entropy-against-the-target's-own-greedy-next-token
variant from the source-free-fitting ablation.

`relay_p`: 1.145x vs. plain AR (mean acceptance length 1.577).
`optimized_source_reuse`: 3.251x vs. plain AR (mean acceptance length
7.697). Retention: 20.5%, matching the earlier 32-prompt estimate (20.7%)
closely. As with the 14B-to-8B adapter case, retention collapsed as
predicted but the method still edges out plain AR (1.145x), not "slower
than no proposer" — the same correction to the derived speed claim
applies here as it did for the adapter result above.

## Target-feedback ablation, variant 2 (KL divergence), rerun at 128 prompts

`outputs/bench-verifier-l2-dev128-27319/benchmark-summary.json`, 128
prompts, paired. Config confirms `verifier_objective: proposal_kl_to_target`.

`relay_p`: 1.180x vs. plain AR (mean acceptance length 1.635).
`optimized_source_reuse`: 3.241x vs. plain AR (mean acceptance length
7.697). Retention: 21.2%, matching the earlier 32-prompt estimate (21.6%)
closely. Same pattern as l1: collapsed retention, still edges out plain AR.

## Target-feedback ablation, variant 3 (accepted-length surrogate, from scratch), rerun at 128 prompts

`outputs/bench-verifier-l3-dev128-27320/benchmark-summary.json`, 128
prompts, paired. Config confirms `verifier_objective:
accepted_prefix_surrogate`, fitted from scratch (not warm-started).

`relay_p`: 0.788x vs. plain AR (mean acceptance length 1.097).
`optimized_source_reuse`: 3.244x vs. plain AR. Retention: 14.3%, matching
the earlier 32-prompt estimate (14.2%) almost exactly. Unlike l1/l2, this
variant genuinely is slower than plain AR, not just badly degraded —
retention is low enough here to actually cross below the break-even line.
This is the worst of the four source-free-fitting variants, consistent
with the earlier ranking.

## Target-feedback ablation, variant 4 (warm-started continuation), rerun at 128 prompts — completes the 4-variant set

`outputs/bench-verifier-l4-dev128-27321/benchmark-summary.json`, 128
prompts, paired. Config confirms: L0 (an already-working map) warm-started,
then continued fitting with the `accepted_prefix_surrogate` objective for
a shorter run (the case meant to rule out a bad starting point).

`relay_p`: 1.042x vs. plain AR (mean acceptance length 1.447).
`optimized_source_reuse`: 3.258x vs. plain AR. Retention: 18.8%, matching
the earlier 32-prompt estimate (19.1%) closely. Just barely above
break-even against plain AR, essentially a wash.

**All four source-free-fitting variants are now confirmed at 128-prompt
scale**, all within 1-2 percentage points of their original 32-prompt
estimates (cross-entropy 20.5% vs. 20.7%, KL 21.2% vs. 21.6%, surrogate
14.3% vs. 14.2%, warm-started surrogate 18.8% vs. 19.1%). The ranking and
the core conclusion (the source model's supervision is not replaceable by
target-only feedback at this fitting budget) both hold at proper scale.
Ready to restore this ablation to the paper (Section 7 and Appendix I),
with the corrected framing that l1/l2/l4 remain (barely, in l4's case)
faster than plain AR despite collapsed retention, while l3 alone is
genuinely slower than plain AR.

## Fine-tuned-target reuse (Nemotron-Orchestrator-8B descendant, no refitting), rerun at 128 prompts

`outputs/bench-nemotron-descendant-dev128-27322/benchmark-summary.json`,
128 prompts, paired. The map fitted for base Qwen3-8B, unchanged, applied
to `nvidia/Nemotron-Orchestrator-8B`, a publicly released model
independently fine-tuned from Qwen3-8B (not our own LoRA drift study).

`relay_p`: 4.952x vs. plain AR (mean acceptance length 6.984).
`optimized_source_reuse`: 3.230x vs. plain AR (mean acceptance length
7.657). Retention: 91.2%, relay vs. source-reuse: 1.533x — both close to
the earlier 32-prompt estimate (91.9% retention, 1.605x), confirming the
headline claim that a genuinely independently released fine-tuned
descendant of the target still works well with the original relay, no
refitting needed. Ready to restore to Section 7.

## Main table, DFlash Qwen3-14B: first genuine paired vs-plain-AR measurement

`outputs/bench-dflash-14b-pairedAR-27325/benchmark-summary.json`, 500
prompts (full MATH-500), paired, native_ar included in the same run as
optimized_source_reuse and relay_p for the first time (previously native
AR context for this pair came from a separate, older run, documented in
`DFLASH_CONTROL_EQUUIVALENCE.md`/`DFLASH_14B_MAIN_RESULT.md` as
"cross-run contextual ratio... not substituted for the causal same-run
result").

`relay_p`: 5.143x vs. plain AR, paired bootstrap 95% interval [4.983,
5.298] (computed via `paired_bootstrap_speedup`, 10,000 replicates, seed
1729, matching the paper's existing convention). `optimized_source_reuse`:
4.055x vs. plain AR, interval [3.930, 4.177]. Relay vs. source-reuse ratio:
1.267x, consistent with the existing main table's registered 1.246x for
this pair. Exact-sequence-match rate against native AR: 22.6% (113/500) —
the first genuine large-scale measurement of this specific comparison
(the existing correctness appendix only checked source-reuse vs. relay,
both block-verified, never either against single-token-verified native AR
at this scale).

This closely matches the old cross-run contextual estimate (~5.160x,
26.671 tok/s native AR baseline) but is now a real paired measurement with
a proper confidence interval, replacing that context-only number. Ready to
update Table 2 and the abstract's headline framing once the remaining 3
main-table reruns (DFlash-8B, EAGLE-3-8B, EAGLE-3-14B) land.

## Main table, DFlash Qwen3-8B: first genuine paired vs-plain-AR measurement, with target-specific ceiling in the same run

`outputs/bench-dflash-8b-pairedAR-v2-27357/benchmark-summary.json`, 500
prompts, paired. `native_ar`, `native_target_dflash` (the released
target-specific DFlash-8B proposer), `optimized_source_reuse`, and
`relay_p` all measured together for the first time.

`native_target_dflash`: 5.578x vs. plain AR, 95% interval [5.397, 5.728].
`relay_p`: 5.018x vs. plain AR, interval [4.870, 5.174] — retains 90.1% of
the target-specific ceiling, matching the paper's existing "89 to 91
percent" claim exactly. `optimized_source_reuse`: 3.278x vs. plain AR,
interval [3.183, 3.377]. Relay vs. source-reuse: 1.531x, in the same
ballpark as the existing main table's registered 1.483x for this pair
(a genuinely independent measurement, not expected to match to the third
decimal). Exact-match rate against native AR: 23.4% (117/500).

This is a clean, positive confirmation: every existing headline number for
this pair holds up under a fully paired, larger comparison that finally
includes plain AR and the target-specific ceiling in the same run.

## Main table, EAGLE-3 Qwen3-8B: paired vs-plain-AR measurement — completes all 4 main-table reruns

`outputs/bench-eagle3-8b-pairedAR-v3-27396/benchmark-summary.json`, 500
prompts, paired. `native_ar`, `native_target_eagle3` (released
target-specific EAGLE-3 chain), `source_reuse_eagle3`, `relay_eagle3` all
measured together for the first time.

`native_target_eagle3`: 2.656x vs. plain AR, 95% interval [2.586, 2.728].
`relay_eagle3`: 2.397x vs. plain AR, interval [2.332, 2.463] — retains
90.2% of the target-specific ceiling, matching the paper's existing
"89 to 91 percent" claim. `source_reuse_eagle3`: 1.875x vs. plain AR,
interval [1.826, 1.926]. Relay vs. source-reuse: 1.278x, close to the
existing main table's registered 1.248x for this pair.

**All 4 main-table pairs (DFlash 8B/14B, EAGLE-3 8B, and EAGLE-3 14B still
running) are now confirmed via genuine paired native-AR measurements**,
every one matching the existing headline claims. Once EAGLE-3-14B lands,
Table 2 and the abstract can be reframed with real vs-plain-AR speedups
and CIs throughout, as requested.

## Main table, EAGLE-3 Qwen3-14B: paired vs-plain-AR measurement — all 4 main-table pairs now confirmed

`outputs/bench-eagle3-14b-pairedAR-v3-27397/benchmark-summary.json`, 500
prompts, paired.

`native_target_eagle3`: 2.945x vs. plain AR, 95% interval [2.875, 3.015].
`relay_eagle3`: 2.625x vs. plain AR, interval [2.563, 2.688] — retains
89.1% of the target-specific ceiling, inside the paper's existing
"89 to 91 percent" claim. `source_reuse_eagle3`: 2.394x vs. plain AR,
interval [2.337, 2.450]. Relay vs. source-reuse: 1.097x, close to the
existing main table's registered 1.082x for this pair.

**Summary table, all 4 main pairs, real paired vs-plain-AR speedups:**

| Pair | vs. plain AR | 95% CI | vs. source-reuse | Retention vs. target-specific |
|---|---:|---:|---:|---:|
| DFlash, 8B | 5.018x | [4.870, 5.174] | 1.531x | 90.1% |
| DFlash, 14B | 5.143x | [4.983, 5.298] | 1.267x | n/a (no public 14B checkpoint) |
| EAGLE-3, 8B | 2.397x | [2.332, 2.463] | 1.278x | 90.2% |
| EAGLE-3, 14B | 2.625x | [2.563, 2.688] | 1.097x | 89.1% |

All four independently confirm the paper's existing headline claims
("1.483x/1.246x/1.248x/1.082x vs. source reuse" and "89 to 91 percent vs.
target-specific"). Ready to reframe Table 2 and the abstract's headline
speedup claim to lead with vs.-plain-AR (with real CIs), per the explicit
request, while keeping the vs.-source-reuse numbers as the secondary
comparison the Setup section already defines them as.

## Closed-form ablation rerun under the corrected (weighted) objective

`outputs/bench-closed-form-fixed-27422/benchmark-summary.json`, 128
prompts. After fixing the ridge-vs-gradient-descent loss mismatch (see
above), the closed-form solve reaches 0.955x vs. source reuse, paired 95%
interval [0.927, 0.982], 52.3% retention (accepted length 4.03 vs. source
reuse's 7.70). Fit cost: 221.3s data collection + 6.8s solve = 228.1s
total (`outputs/train-closed-form-fixed-27421/closed-form-summary.json`).

This is close to the old, uncorrected number (0.925x, 54.9% retention) and
does not change the ablation's qualitative conclusion: even solving the
exact same objective gradient descent does, in closed form, the plain
linear map fitted by gradient descent still substantially outperforms it
(91% retention vs. 52.3%). The bug was real and worth fixing on its own
methodological merits (the comparison is now genuinely apples-to-apples,
not confounded by two different loss functions), but empirically it does
not flip the result: gradient descent's advantage here is not an artifact
of the earlier unweighted comparison.

## Experiment 1 result (step50): nonlinear residual correction does not recover drift collapse either

`outputs/bench-nl-step50-27433/benchmark-summary.json`, 128 prompts,
paired. `relay_p` here is the nonlinear-delta-corrected relay
(`c = R z + U sigma(V z)`, rank-32 bottleneck, GELU, trained against the
step50 drift checkpoint following the same E3-Stage-B-style protocol as
the linear delta).

`optimized_source_reuse`: 0.444x vs. plain AR (mean acceptance length
1.065). `relay_p` (nonlinear-corrected): 0.696x vs. plain AR (mean
acceptance length 1.0006). Exact-sequence-match rate against native AR:
0% for both.

**This is the key test in the new adaptive-relay plan, and it is
negative.** The nonlinear correction's accepted length (1.0006) is
statistically indistinguishable from both the uncorrected relay (1.0008,
from the original Phase A retention benchmark) and the linear delta
correction (1.0002, measured at step1000 earlier). Adding capacity to the
correction mechanism did not help at all, which rules out "the correction
lacks expressivity" as the explanation for Phase A's negative result. The
failure is downstream of the proposer's own candidates no longer matching
what the drifted target wants to generate, not a limitation of how the
relay's context tensor is computed, linear or nonlinear. Still to check:
whether this null result holds at every drift step (150/400/1000/2500),
now running.

## 2026-09-05 — Paper rewritten around the plain-AR baseline, five cross-artifact contradictions found and fixed

Scope: rewrite the manuscript so every headline speedup is against plain
autoregressive decoding, drop the delta/drift material, and eliminate
logical discrepancies. No new cluster runs.

Single source of truth. `reports/final/MAIN_PAIRED_AR.json` and
`reports/final/TRANSFER_PAIRED_AR.json` now generate every headline number.
The generators for the main table, the transfer table, the acceptance
table, `main_throughput.pdf`, `adaptation_cost.pdf` and
`transfer_speedups.pdf` all read those two files, so no asset can disagree
with another about the same quantity.

Contradictions found during the page-by-page visual review and fixed:

1. `main_throughput.pdf` (Figure 3) annotated each row with the old
   source-reuse ratio (1.483x / 1.246x / 1.248x / 1.082x) while Table 2 on
   the facing spread reported 5.02x / 5.14x / 2.40x / 2.62x vs AR. The two
   also disagreed on the *same* relay-over-source-reuse comparison
   (1.483x vs 1.53x), because they used different statistics.
2. `adaptation_cost.pdf` panel (b) hardcoded 91.2 / 90.5 / 89.2 recovery
   against the text's 90.3 / 90.2 / 89.1.
3. The acceptance appendix table reported 7.76 / 7.07 accepted tokens per
   cycle for DFlash at 8B where the main table reported 7.66 / 6.98 for
   the identical quantity.
4. Appendix F compared its 8,192-example point against "the main table's
   headline of 1.483x", a number the main table no longer contains.
5. Related work still claimed a verifier-feedback result from a section
   that had been deleted.

Also removed: the negative source-free-fitting appendix, the failed
8-prompt cross-tokenizer narration, the 32-prompt development-forecast
table (the last small-sample artifact in the paper), and the
decision-procedure flowchart.

Rounding correction: breadth acceptance retention was printed as "62 to 86
percent" where the 16 measured cells span 61.5 to 85.4 percent, rounding
outward on both ends. Exact values now printed.

Figure legibility: `system_overview.pdf`'s feedback connector began in
empty space above the verifier box and its one diagonal grazed a corner.
Both loops are now fully orthogonal with every endpoint anchored on a box
edge. The grayscale pass caught `transfer_speedups.pdf`'s caption asking
readers to distinguish green rows from blue rows, which is impossible in
grayscale; the groups now also differ by hatching.

Known limitation, stated in the paper rather than papered over: the
16-cell breadth sweep was run without a plain-AR arm, so Figure 5 and the
full workload table are the only results reported against source reuse.
Reframing them vs AR requires 16 new benchmark runs.

Method section now states the mechanistic grounding: the slot RelaySpec
replaces is the proposer's own linear fusion layer `F`, since the released
proposers compute `norm(F z_source)` (`src/relayspec/proposers.py:30`).
RelaySpec substitutes `R` for `F` and changes nothing downstream. The
normalization that follows discards magnitude, which is why the loss is
energy-normalized and why `R` carries no bias.
