# RelaySpec final execution runbook

## Paper question

Can a frozen feature-conditioned speculative proposer trained for a 4B model be
reused with an 8B or 14B verifier by translating verifier features directly
into the proposer's consumed interface, eliminating the repeated 4B source
trunk while retaining exact target verification?

The study answers this for two structurally different released proposers:
DFlash (parallel diffusion proposals) and the pinned DeepSpec EAGLE-3
length-7 autoregressive-chain evaluator. The primary comparison is always
against an optimized source-reuse
baseline, because that is the computation RelaySpec claims to remove.

## Narrow novelty claim

RelaySpec does not claim that verifier hidden states, pretrained-drafter
steering, generic drafters, parameter-efficient adaptation, or KV reuse are new.
Those spaces are covered respectively by KVShot, SD-squared, PARD/OmniDraft,
EDA, and related cache-transfer work. The paper's narrower contribution is:

> transplant the already learned conditioning interface of an unchanged,
> frozen feature-conditioned proposer to a new compatible verifier using one
> per-target affine map, specifically to eliminate a separately executed source
> transformer from every speculative cycle.

The cross-mechanism demonstration is essential: DFlash and the pinned DeepSpec
EAGLE-3 evaluator consume different interface geometries and generate proposals
differently, yet the same interface-transplantation principle and source-removal
cost model apply. DeepSpec's released evaluator uses the checkpoint's length-7
autoregressive chain; this experiment does not claim the dynamic-tree numbers
reported by the separate SafeAILab/AngelSlim serving implementations. The
novelty and source-status audit is recorded in
`docs/research/relayspec-source-log.md`; preprints and technical notes are never
presented as peer-reviewed evidence.

## Frozen mechanism

1. Prefill the full target and retain its exact KV cache.
2. Read five target hidden-state taps already produced by verification.
3. Apply one target-specific affine relay into the exact frozen-proposer
   context tensor. DFlash normalizes because its consumed tensor is normalized;
   EAGLE-3 preserves input scale because its consumed tensor is raw.
4. Let the unchanged frozen proposer generate its normal DFlash block or
   DeepSpec EAGLE-3 length-7 autoregressive-chain proposal.
5. Verify all proposed positions with the full target, commit only the canonical
   accepted prefix plus target correction, crop speculative cache entries, and
   repeat.

The relay is approximate; token selection is exact with respect to the target
verification path. Relay error can reduce acceptance and speed, but cannot
approve a wrong token.

## Why every fixed choice exists

| Choice | Value | Backing |
|---|---:|---|
| target authority | full verifier only | correctness derivation and Leviathan et al. verification rule |
| model mode | Qwen3 non-thinking, greedy | DFlash ICML 2026 evaluation setup; enables deterministic paired sequence checks |
| output cap | 2,048 | DFlash setup and direct paper comparability; observed 8--12% MATH cap-hit rates are reported, so the cap is not misrepresented as truncation-free |
| DFlash block | 16 | released checkpoint contract plus completed 8/16/32 throughput ablation |
| EAGLE draft cap | 7 | pinned released `ttt7` checkpoints and official DeepSpec path |
| source layers | 0--33 | tap 33 is the last consumed source feature; later layers have no causal path to proposer context |
| five taps | released indices | immutable proposer checkpoint contract, not a new tuned hyperparameter |
| loss | relative interface MSE | coefficient-free radial/angular identity `rho^2 + 1 - 2 rho cos(theta)` |
| cosine/KL weights | none | no defensible mixing coefficient; historical `0.1` refinements did not resolve a benefit |
| EAGLE input scale | preserved | raw-interface derivation plus matched live ablation: 1.237x vs 1.035x |
| learning rate | 6e-4 | official EAGLE precedent and matched 128-step pilot: 0.413 vs 0.586 loss |
| weight decay / clipping | 0 / 1 | pinned EAGLE training configuration |
| fit budget | 4,096 examples, 1,024 steps, length 192 | controlled cross-family compute budget; explicitly not claimed globally optimal |
| bootstrap | paired requests, 10,000 draws, seed 1729, 95% | preserves prompt/output coupling; 2.5% tail Monte Carlo SE is about 0.00156 |

The machine-audited source of truth is `configs/relayspec_protocol.yaml`.

## Baselines and causal comparisons

Each proposer/target pair has four methods:

1. **Native AR:** one-token target decoding.
2. **Native proposer:** released target-specific DFlash or EAGLE-3.
3. **Optimized source reuse:** 4B proposer plus source layers 0--33, executed
   only for committed positions after target verification.
4. **RelaySpec:** the same frozen 4B proposer with the source trunk removed.

The main causal estimate is RelaySpec versus optimized source reuse. Native AR
and native target-specific proposer establish the broader deployment frontier.
At 14B, incompatible model groups are loaded in separate jobs to avoid changing
timings through memory pressure; source/relay remain paired in one process and
native AR/native proposer remain paired in another.

## Experiment matrix

### Main table: full MATH-500 plus independent complement

| Proposer family | Source | Target | Methods | Prompts | Status |
|---|---|---|---|---:|---|
| DFlash | Qwen3-4B | Qwen3-8B | all four | 500 | historical complete; unified artifact regeneration |
| DFlash | Qwen3-4B | Qwen3-14B | AR/source/relay; native if released checkpoint is available | 500 | historical complete; unified artifact regeneration |
| EAGLE-3 | Qwen3-4B | Qwen3-8B | all four in two paired groups | 500 | queued as jobs 25561 and 25563 |
| EAGLE-3 | Qwen3-4B | Qwen3-14B | all four in two paired groups | 500 | queued as jobs 25562 and 25564 |

All use the same immutable MATH-500 manifest, prompt formatter, seed, 2,048 cap,
bf16, SDPA backend, four RTX 6000 Ada GPUs, and node. Method order rotates by
request to reduce thermal/order bias. Because the 32 architecture-development
prompts came from MATH-500, the conventional 500-question table is reported for
paper comparability but is not mislabeled fully untouched. Confirmatory claims
are additionally computed on the fixed 468-question complement obtained by
normalized prompt hashes before the run. Its manifest SHA-256 is
`c2da7d7822f7d57d1ccca542d4645ce74b98f7e748c60db8377691f8bbf8486b`.

The four GPUs are four independent data-parallel evaluation ranks, not tensor
parallel shards of one request. Reported tokens/s is total generated tokens
divided by the sum of per-request latencies, so it is a micro-averaged
single-request rate rather than an incorrectly four-times-larger cluster
throughput. Four-way sharding reduces experiment wall time without changing
the statistic.

### Breadth table after the MATH freeze

Run source reuse and RelaySpec at both target scales and both proposer families:

| Task | Count | Official metric | Purpose |
|---|---:|---|---|
| GSM8K deterministic subset | 128 | exact final answer | short mathematical reasoning |
| HumanEval | 164 | pass@1 | code generation |
| EvalPlus MBPP | 378 | base/plus pass@1 | stronger code tests |
| MT-Bench | 80 two-turn questions | speed, acceptance, and paired output agreement | open-ended and multi-turn behavior; no quality-judge claim without a separately executed judge pipeline |

AIME is excluded by scope. MATH-500 remains the primary result; the breadth
table tests whether acceptance/speed behavior transfers across output styles.

## Measurements retained per request

- request latency, decode time, output tokens, throughput, and peak memory;
- target verification, proposer, source trunk, relay, prefill, and unattributed
  CUDA-region time;
- proposal length, accepted length by position, verification calls, draft calls,
  and committed source tokens;
- completion text and token hash for exact paired agreement;
- official task score, not a proxy score;
- checkpoint size, parameter count, fit wall time, and four-GPU hours.

Primary speed is the ratio of summed paired reference time to summed paired
candidate time. Report the point estimate and paired cluster-bootstrap 95%
interval: one request per cluster except that both dependent MT-Bench turns are
resampled as one conversation. Also report median/p95 latency, tokens/s, acceptance, memory, and
component percentages. Task accuracy uses the official scorer and a paired
example bootstrap interval. No fixed cosine/error threshold participates in
token commitment or model selection.

The inherited backends currently emit a zero placeholder for TTFT rather than a
synchronized first-token event. TTFT is therefore marked unavailable and
excluded from claims; zero is never presented as a measurement.

## Required ablations (small and causal)

1. **Remove the relay:** optimized source reuse versus RelaySpec. This is the
   headline causal intervention.
2. **Respect versus erase interface scale:** normalized versus scale-preserving
   EAGLE relay, same parameters and training. Completed result: 1.035x versus
   1.237x on the development split.
3. **Remove source-only waste:** naive full source execution versus committed-
   prefix layers 0--33. This proves the baseline is not intentionally weak.
4. **Mechanism/scale:** DFlash versus EAGLE-3 and 8B versus 14B. This tests the
   Amdahl prediction that gain tracks removable source share.

Rank sweeps, arbitrary loss mixtures, and many tap-count variants are omitted:
relay compute is already below 0.5% of runtime and the released proposer fixes
the five-tap interface, so these sweeps have little causal or systems value.

## Gates and interpretation

- **Correctness gate:** only full-target verification may commit tokens, cache
  conformance tests must pass, and every observed sequence disagreement is
  retained and audited. Exact source/relay sequence agreement is reported as a
  reproducibility diagnostic, not substituted for the formal target-authority
  argument: different accepted block segmentations can select different BF16
  kernels at near-tied logits. Official paired quality must differ by at most
  the preregistered one percentage-point study tolerance.
- **Speed gate:** paired speed CI lower bound must exceed 1.0.
- **Mechanism gate:** the development-derived Amdahl forecast must agree with
  untouched-test speed within profiling uncertainty, and gain must covary with
  removable source share across scale/family. A same-run component calculation
  is labeled a reconstruction, not misrepresented as an independent forecast.
- **Quality gate:** RelaySpec and source reuse must have identical official task
  outcomes whenever their sequences agree; all disagreements are scored and
  the paired accuracy delta and interval are reported. Comparisons to native
  AR/native proposer remain separate because block and token kernels can
  diverge at floating-point ties.
- **Generalization gate:** positive source-relative speed at both 8B and 14B
  for both proposer families on the primary task.

Passing these gates supports the narrow paper claim: **interface
transplantation converts otherwise redundant source-model work into a cheap
target-specific adapter, and its end-to-end benefit is predictable from
removable runtime share and retained acceptance.** It does not claim a universal
speedup over every target-specific proposer.

## Serialized execution order on exactly four GPUs

1. Complete 8B and 14B development confirmations. **Done:** 1.237x and 1.086x,
   both 32/32 exact.
2. Run EAGLE-3 full MATH source/relay at 8B and 14B (jobs 25561--25562),
   reporting both all 500 and the predeclared 468-question complement.
   **Done:** 1.250x at 8B and 1.081x at 14B on the complement, with zero
   official accuracy delta.
3. Run EAGLE-3 full MATH native pairs at 8B and 14B (jobs 25563--25564).
   **Done:** target-specific EAGLE reaches 2.567x native AR at 8B and 2.944x at
   14B; RelaySpec retains 90.46% and 89.20% of those respective ceilings.
4. Fit clean coefficient-free DFlash relays at 8B/14B, pass their 32-prompt
   development gates, then run unified DFlash MATH jobs 25565--25566. The
   validation Slurm step now exits nonzero unless all 32 pairs are present,
   source/relay sequences agree exactly, the observed speed and its paired
   bootstrap lower bound both exceed 1.0, and acceptance retention exceeds the
   measured Amdahl break-even. Because later jobs use `afterok`, this is an
   executable gate rather than a manual intention. The already-submitted 8B
   job had snapshotted the earlier batch script, so its identical gate was
   recomputed immediately from the immutable rows and passed at 1.523x
   [1.488, 1.558], 32/32 exact. The pending 14B validation was replaced by
   gated job 25605 and all dependencies were rewired before execution. Job
   25565 is additionally
   held until both target scales can be compared on development evidence. Job
   25599 showed that the older 14B historical artifact is fast, but it differs
   in training history and therefore cannot identify the effect of its loss.
   Jobs 25606--25609 now refit both targets with the same data, steps, seed,
   optimizer, architecture, and taps, changing only relative interface MSE to
   historical MSE plus `0.1` cosine distance. Before seeing these matched
   results, the selection rule was frozen as the higher paired-speed candidate
   at each target scale, conditional on passing exactness, confidence-bound,
   acceptance-retention, and Amdahl gates. Thus `0.1` is a tested historical
   candidate rather than an unexplained final-method constant. **Done:**
   coefficient-free relative MSE wins at both scales (1.523x versus 1.509x at
   8B; 1.265x versus 1.251x at 14B). The direct paired candidate intervals also
   exclude one, so no cosine coefficient or conditional neighbor sweep enters
   the final method. Checkpoint hashes and raw evidence are frozen in
   `reports/design-selection/dflash/OBJECTIVE_SELECTION.md` before releasing
   full evaluation.
5. Score MATH with the pinned Qwen math evaluator and regenerate bootstrap,
   Amdahl, memory, and accuracy tables from immutable JSONL. **Done:** DFlash
   reaches 1.483x [1.475, 1.492] at 8B and 1.246x [1.238, 1.253] at 14B,
   with exact source/relay sequences and identical official accuracy on all
   1,000 prompts.
6. Run the four breadth tasks, largest evidential value per GPU-hour first:
   GSM8K-128, HumanEval, MBPP, then MT-Bench. **In progress:** EAGLE-3 8B
   math/code generation and official scoring are complete with positive paired
   speed intervals and zero paired quality delta. Its first combined artifact
   exposed a driver completeness bug: only turn zero of each MT-Bench record
   ran. The math/code rows remain valid; the chat row is excluded. The fixed
   iterator maintains per-method history and emits both turn-specific IDs.
   Dedicated jobs 25613/25614 produce complete 160-turn EAGLE chat cells after
   the existing serialized chain, with scorer jobs 25615/25616. The matrix
   builder requires 160 requests grouped into 80 bootstrap clusters and
   replaces only the incomplete EAGLE chat summaries.
7. Run isolated source-only and relay-only EAGLE residency probes at both
   scales (jobs 25595--25598). This avoids misreading peak memory from the
   paired process, which intentionally keeps both providers resident.
8. Regenerate the claim-evidence map, final tables/figures, and LaTeX report;
   audit every number back to a raw artifact and pinned source.

No two RelaySpec GPU jobs overlap. CPU scoring follows completed generation and
does not alter model outputs.
