# RelaySpec final evidence ledger

Last updated: 2026-08-28. All primary throughput values are request-level
end-to-end output tokens per second. Every paired GPU generation job used
exactly four RTX 6000 Ada GPUs, batch one, BF16, dense SDPA, greedy Qwen3
non-thinking generation, and a 2,048-token cap.

## Main finding

RelaySpec transplants an already-trained feature-conditioned speculative
proposer to a new target scale. A target-specific bias-free linear relay maps
five hidden-state taps already computed by the target verifier into the exact
consumed interface of the frozen proposer. This removes the old Qwen3-4B
conditioning trunk. The target verifier remains the only token-commit
authority, so relay error affects proposal acceptance and runtime rather than
granting the relay authority over output tokens.

The same mechanism is implemented for two different proposal processes:
DFlash block diffusion and a pinned DeepSpec EAGLE-3 autoregressive chain. It
is evaluated with Qwen3-8B and Qwen3-14B targets.

## Primary MATH-500 evidence

| Family | Target | Source tok/s | Relay tok/s | Relay/source [95% CI] | Exact source/relay | Official accuracy S/R |
|---|---:|---:|---:|---:|---:|---:|
| DFlash | 8B | 148.332 | 219.991 | **1.4831x [1.4745, 1.4917]** | 500/500 | 74.2% / 74.2% |
| DFlash | 14B | 110.450 | 137.604 | **1.2458x [1.2382, 1.2533]** | 500/500 | 79.6% / 79.6% |
| EAGLE-3 | 8B | 82.827 | 103.371 | **1.2480x [1.2420, 1.2540]** | 496/500 | 73.2% / 73.2% |
| EAGLE-3 | 14B | 64.969 | 70.268 | **1.0816x [1.0763, 1.0868]** | 498/500 | 79.6% / 79.6% |

The independent 468-question MATH complement excludes the 32 registered
design-selection prompts. EAGLE-3 produces 1.2495x [1.2435, 1.2555] at 8B and
1.0813x [1.0759, 1.0866] at 14B, with identical 72.863% and 79.487% paired
official accuracies. Every finite-precision mismatch is retained in an audit;
none changes the official MATH outcome.

## Frozen cross-task matrix

The table below uses one frozen checkpoint per family/target across GSM8K,
HumanEval, EvalPlus MBPP, and both MT-Bench turns. MT-Bench's two turns are one
bootstrap cluster. MT-Bench has no judge-quality claim.

| Family | Target | Task | N/clusters | Source tok/s | Relay tok/s | Speedup [95% CI] | Exact |
|---|---|---|---:|---:|---:|---:|---:|
| EAGLE-3 | 8B | GSM8K | 128/128 | 85.987 | 108.453 | **1.2613x [1.2437, 1.2789]** | 99.22% |
| EAGLE-3 | 8B | HumanEval | 164/164 | 71.747 | 81.099 | **1.1304x [1.1172, 1.1434]** | 96.95% |
| EAGLE-3 | 8B | MBPP | 378/378 | 69.855 | 82.103 | **1.1753x [1.1656, 1.1850]** | 97.62% |
| EAGLE-3 | 8B | MT-Bench | 160/80 | 42.273 | 49.451 | **1.1698x [1.1501, 1.1906]** | 89.38% |
| EAGLE-3 | 14B | GSM8K | 128/128 | 67.807 | 73.407 | **1.0826x [1.0675, 1.0979]** | 100.00% |
| EAGLE-3 | 14B | HumanEval | 164/164 | 55.384 | 50.577 | 0.9132x [0.9011, 0.9255] | 99.39% |
| EAGLE-3 | 14B | MBPP | 378/378 | 54.549 | 50.492 | 0.9256x [0.9151, 0.9361] | 98.68% |
| EAGLE-3 | 14B | MT-Bench | 160/80 | 33.828 | 34.575 | 1.0221x [0.9999, 1.0449] | 96.88% |
| DFlash | 8B | GSM8K | 128/128 | 116.869 | 166.388 | **1.4237x [1.4040, 1.4445]** | 100.00% |
| DFlash | 8B | HumanEval | 164/164 | 118.844 | 144.999 | **1.2201x [1.2017, 1.2383]** | 100.00% |
| DFlash | 8B | MBPP | 378/378 | 117.772 | 151.554 | **1.2868x [1.2732, 1.3001]** | 100.00% |
| DFlash | 8B | MT-Bench | 160/80 | 60.851 | 81.616 | **1.3412x [1.3183, 1.3639]** | 100.00% |
| DFlash | 14B | GSM8K | 128/128 | 88.045 | 97.072 | **1.1025x [1.0836, 1.1214]** | 100.00% |
| DFlash | 14B | HumanEval | 164/164 | 87.337 | 77.609 | 0.8886x [0.8741, 0.9038] | 100.00% |
| DFlash | 14B | MBPP | 378/378 | 88.020 | 83.977 | 0.9541x [0.9416, 0.9663] | 100.00% |
| DFlash | 14B | MT-Bench | 160/80 | 45.319 | 49.295 | **1.0877x [1.0653, 1.1083]** | 100.00% |

Raw RelaySpec has a 1.1135x equal-cell geometric mean and 11/16 strictly
positive paired intervals. The negative 14B code cells are included rather
than discarded.

## Automatic no-slowdown policy

Let the optimized source latency be decomposed as

```text
source = cycle-dependent proposal/verification + removable source trunk + other
relay  = cycle-dependent work adjusted for acceptance + relay + other.
```

With source fractions `p_C + p_S + p_O = 1`, relay fraction `p_R`, and
source/relay accepted-token yields `a_S` and `a_R`, the measured model is

```text
L_R / L_S = (a_S / a_R) p_C + p_O + p_R.
```

The relay is eligible only when this predicts speedup above one and the paired
95% speed lower bound also exceeds one. Otherwise the system retains source
reuse. These are the algebraic no-slowdown and statistical-evidence boundaries,
not tuned similarity constants.

The Amdahl prediction matches the observed point-estimate direction in 16/16
cross-task cells with 0.34% mean absolute relative error. The conservative
policy chooses source for both 14B code tasks and the inconclusive EAGLE-14B
MT-Bench cell. Its descriptive equal-cell geometric mean is 1.1354x and it
selects no measured slowdown. This is a calibrated workload-level policy; it
is not presented as zero-cost per-request oracle routing.

## Official quality

Within every scored family/target/task pair, source and relay have identical
official quality:

| Family | Target | GSM8K | HumanEval base / Plus | MBPP base / Plus |
|---|---:|---:|---:|---:|
| DFlash | 8B | 93.75% | 85.98% / 80.49% | 84.13% / 73.02% |
| DFlash | 14B | 94.53% | 89.02% / 83.54% | 88.36% / 75.13% |
| EAGLE-3 | 8B | 92.97% | 87.20% / 82.32% | 83.07% / 71.96% |
| EAGLE-3 | 14B | 94.53% | 90.24% / 85.98% | 88.89% / 74.87% |

MATH uses the pinned Qwen math scorer. HumanEval and MBPP use pinned EvalPlus
revision `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`. Exact text agreement is an
additional finite-precision diagnostic, not a substitute for the target-
authority argument or official task scoring.

## Native and target-specific controls

| Family | Target | Native AR | Native target proposer | Source | Relay | Relay/native proposer |
|---|---:|---:|---:|---:|---:|---:|
| DFlash | 8B | 44.52 | 241.25 | 148.33 | 219.99 | 91.19% |
| DFlash | 14B | 26.71 | unavailable | 110.45 | 137.60 | unavailable |
| EAGLE-3 | 8B | 44.45 | 114.28 | 82.83 | 103.37 | 90.46% |
| EAGLE-3 | 14B | 26.71 | 78.78 | 64.97 | 70.27 | 89.20% |

Native target-specific proposers remain the throughput frontier. RelaySpec's
claim is a much cheaper portability path that recovers 89--91% of available
target-specific proposer throughput while avoiding proposer retraining.

## Memory and adaptation cost

| Target | EAGLE source peak | Relay peak | Peak saving | Steady allocated saving |
|---|---:|---:|---:|---:|
| 8B | 25.563 GiB | 17.763 GiB | **7.800 GiB / 30.51%** | 7.448 GiB / 30.35% |
| 14B | 37.565 GiB | 29.938 GiB | **7.627 GiB / 20.30%** | 7.422 GiB / 20.17% |

The relays contain 52--66M parameters. A 4,096-example, 1,024-update fit takes
86.7--118.5 seconds on four GPUs. The data volume is approximately 195x below
DFlash's reported 800K target-generated examples and 130x below EAGLE-3's
approximately 532K entries. These are data-volume comparisons; no
cross-hardware training-cost ratio is inferred.

The relay occupies 0.38--0.72% of reference runtime in the cross-task matrix,
whereas the removed source trunk occupies 26.8--40.0%. A low-rank relay cannot
provide a material end-to-end gain at these shares; interface fidelity and
accepted-token retention are the relevant optimization variables.

## Minimal design evidence

- **Objective:** coefficient-free relative interface MSE beat the matched
  historical `MSE + 0.1 cosine` candidate at both target scales. Direct paired
  ratios were 1.0085x [1.0005, 1.0174] at 8B and 1.0114x [1.0046, 1.0177] at
  14B. The conditional `0.03/0.3` bracket was therefore not triggered.
- **EAGLE scale rule:** preserving feature scale improved a matched 8B
  development result from 1.035x to 1.237x and acceptance retention from
  69.0% to 82.6%. The rule was then frozen and transferred unchanged to 14B.
- **DFlash block:** block 16 maximized absolute throughput in the registered
  8/16/32 ablation and matches the released checkpoint configuration.
- **Full rank:** the unrestricted single affine map is the minimal linear
  hypothesis without an arbitrary rank bottleneck; relay arithmetic is already
  below 0.8% of runtime.
- **Data separation:** an operator-preserving normalized-hash audit finds zero
  exact overlap between 4,096 fitting records and all 1,250 evaluation records.

Every decision and numerical constant is classified as formal, peer-reviewed,
official-artifact, development-selected, test-observed, or administrative in
`docs/research/relayspec-decision-register.md` and the machine-audited protocol.

## Verification and scope

- Local suite: 118 tests after final matrix-policy coverage, plus lint,
  protocol-constant audit, prompt-overlap audit, and patch-integrity check.
- Remote proof: final job 25642 passed 118/118 tests on an allocation exposing
  exactly four RTX 6000 Ada GPUs.
- Every result directory includes configuration, manifest digest, rank rows,
  allocation proof, GPU telemetry, scorer hashes, and source revisions.
- Timing claims are scoped to matched batch-one BF16 PyTorch/SDPA inference.
  Production SGLang/vLLM concurrency is not claimed.

## Artifact map

- Full final matrix: `reports/final/BREADTH_MATRIX.md` and `.json`
- DFlash main runs: `reports/final/dflash-{8b,14b}-math500/`
- EAGLE main runs: `reports/final/eagle3-{8b,14b}-math500/`
- Breadth runs: `reports/final/{dflash,eagle3}-{8b,14b}-breadth/`
- Corrected EAGLE chat: `reports/final/eagle3-{8b,14b}-mtbench-two-turn/`
- Memory: `reports/final/EAGLE3_{8B,14B}_MEMORY.md`
- Remote verification: `reports/final/remote-4gpu-verification/`
- Adapter cost: `reports/adapter-cost.md`
- Decision register: `docs/research/relayspec-decision-register.md`
- Primary-source log: `docs/research/relayspec-source-log.md`
- Paper PDF: `output/pdf/RelaySpec_End_to_End_Research_Report.pdf`

## Publication assessment

The evidence supports a focused paper on frozen-proposer interface
transplantation and critical-path source-trunk elimination. The strongest
result is not that a linear map is expressive in isolation; it is that the
same minimal intervention works for diffusion and autoregressive-chain
proposers, predicts its own workload boundary, removes 20--31% peak memory,
fits in under two minutes, and produces 8--48% main-task speedups without an
official accuracy loss.

The result is credible for ICLR submission. Its principal review risk is
novelty positioning relative to recent feature-steering, draft-adaptation, and
target-interface work. The paper must claim the narrow systems contribution
precisely and must not imply universal raw speedup or production-serving
throughput.
