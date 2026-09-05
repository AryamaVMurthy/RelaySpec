# RelaySpec execution status

Last updated: 2026-08-28 after completion of the serialized four-GPU pipeline.

## Status

**All planned training, main evaluation, breadth evaluation, memory, native
control, scoring, mismatch-audit, and verification jobs are complete.** No GPU
job is running or pending.

The final immutable result matrix is
`reports/final/BREADTH_MATRIX.{json,md}`. The complete evidence ledger is
`reports/FINAL_RESULTS.md`.

## Completed experimental scope

- Proposer families: DFlash block-16 diffusion and pinned DeepSpec EAGLE-3
  length-7 autoregressive chain.
- Source proposer scale: Qwen3-4B.
- Target verifiers: Qwen3-8B and Qwen3-14B, non-thinking.
- Primary benchmark: all 500 MATH-500 questions, plus a 468-question
  independent complement excluding the 32 design prompts.
- Breadth: GSM8K-128, HumanEval-164, EvalPlus MBPP-378, and both turns of 80
  MT-Bench conversations.
- Controls: native one-token AR and released target-specific proposer where
  available.
- Resources: separate-process source/relay memory at both EAGLE target scales.
- Design evidence: objective, interface scaling, DFlash block length,
  provider-gate, adaptation time/data/parameters, prompt-overlap audit.

All GPU work used exactly four RTX 6000 Ada GPUs and was serialized so no two
GPU experiments overlapped.

## Final top-line result

| Family | Target | MATH source | MATH relay | Paired speedup | Accuracy S/R |
|---|---:|---:|---:|---:|---:|
| DFlash | 8B | 148.332 | 219.991 | 1.4831x [1.4745, 1.4917] | 74.2% / 74.2% |
| DFlash | 14B | 110.450 | 137.604 | 1.2458x [1.2382, 1.2533] | 79.6% / 79.6% |
| EAGLE-3 | 8B | 82.827 | 103.371 | 1.2480x [1.2420, 1.2540] | 73.2% / 73.2% |
| EAGLE-3 | 14B | 64.969 | 70.268 | 1.0816x [1.0763, 1.0868] | 79.6% / 79.6% |

Across the 16 cross-task family/target/task cells, raw RelaySpec has 1.1135x
geometric-mean speed, 11/16 strictly positive paired intervals, and 16/16
Amdahl direction matches at 0.34% mean absolute relative error. Conservative
provider selection gives 1.1354x descriptive geometric-mean speed and keeps
source reuse for the measured 14B code slowdowns and the statistically
inconclusive EAGLE-14B MT-Bench cell.

## Corrected two-turn EAGLE chat runs

An earlier breadth artifact executed only turn zero of MT-Bench. The driver was
corrected to preserve separate source/relay conversation histories and execute
both turns. The incomplete 80-turn chat rows are excluded from the final
matrix.

- Job 25634, EAGLE-8B: 160 turns / 80 clusters, 1.1698x
  [1.1501, 1.1906], 143/160 exact.
- Job 25635, EAGLE-14B: 160 turns / 80 clusters, 1.0221x
  [0.9999, 1.0449], 155/160 exact. The lower interval does not exceed one, so
  the conservative provider keeps source reuse.
- CPU scorers 25636 and 25637 completed. MT-Bench is not assigned a judge
  quality score.

## Memory result

- EAGLE-8B source/relay peak: 25.563/17.763 GiB; saving 7.800 GiB (30.51%).
- EAGLE-14B source/relay peak: 37.565/29.938 GiB; saving 7.627 GiB (20.30%).

The first memory attempt failed before producing rows because the DeepSpec
launcher omitted the pinned Transformers 5.10.2 overlay. A regression test was
added and the launcher now fails fast without the overlay. A second pre-row
failure exposed a wrong 14B checkpoint pathname; config provenance tests now
validate both memory checkpoints against the promoted main configs. Only the
successful replacement artifacts enter the final report.

## Verification status

- Local: 118 tests, Ruff, protocol audit over 27 active configs,
  operator-preserving prompt-overlap audit, and `git diff --check`.
- First remote verification job 25638: allocation check passed but the suite
  correctly failed because a protocol-required report had not been synced.
- Replacement job 25639 first proved the synchronized required artifacts.
  After adding the final conservative-policy regression, job 25642 reran the
  exact final tree and completed in nine seconds with 118/118 tests on an
  allocation exposing exactly four RTX 6000 Ada GPUs. The immutable final
  allocation and pytest logs are in
  `reports/final/remote-4gpu-verification/`.

## Evidence provenance

The protocol registers every numeric configuration leaf as one of:

- formal/architecturally forced;
- matched to a peer-reviewed primary evaluation;
- copied from a pinned official artifact;
- selected on disjoint development data;
- immutable test evidence; or
- administrative reproducibility budget with an explicit rationale.

The executable constant audit passes. The fit/evaluation overlap audit finds
zero normalized exact overlaps between 4,096 fit records and 1,250 evaluation
records.

## Final artifact locations

- `reports/final/BREADTH_MATRIX.md`
- `reports/FINAL_RESULTS.md`
- `reports/COMPLETION_AUDIT.md`
- `reports/claim-evidence-map.md`
- `reports/amdahl-analysis.md`
- `docs/research/relayspec-decision-register.md`
- `docs/research/relayspec-source-log.md`
- `docs/plans/2026-08-28-relayspec-rigorous-evidence-and-final-benchmarks.md`
- `output/pdf/RelaySpec_End_to_End_Research_Report.pdf`

The only intentionally unclaimed experiment class is production serving
concurrency in vLLM/SGLang. All reported speed numbers are matched batch-one
BF16 PyTorch/SDPA measurements; no production-load extrapolation is made.
