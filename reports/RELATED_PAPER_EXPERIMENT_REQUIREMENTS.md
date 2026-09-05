# Primary-paper evidence behind the RelaySpec experiment design

This table records what each source contributes to the design. It does not use
one paper's hardware speed as a RelaySpec prediction, and preprints never back
correctness-critical choices.

| Primary source | Venue/status | Experiment or analysis requirement it motivates | RelaySpec artifact |
|---|---|---|---|
| DFlash | ICML 2026 accepted | Qwen3 non-thinking; greedy block decoding; 2,048-token cap; math/code/chat tasks; acceptance and end-to-end speed | frozen generation config, MATH-500, GSM8K, HumanEval, MBPP, MT-Bench, cap-hit rate |
| HyperDFlash | arXiv 2026-06; not peer reviewed | exact target-side interface boundary; math/code/chat breadth; accepted length and speed; compact reducer comparison | explicit frozen-proposer distinction, interface-boundary derivation, source/relay component timing, adaptation cost, and breadth matrix |
| AngelSpec | arXiv 2026-07; not peer reviewed | task-dependent acceptance/cost behavior; end-to-end latency and serving-load distinction; online cost models | task breadth, paired Amdahl reconstruction, batch-one scope, and no serving-throughput extrapolation |
| EAGLE-3 | NeurIPS 2025 | second feature-conditioned proposer; multiple tasks/models; acceptance, speed, ablations, data/training cost | pinned DeepSpec length-7 chain at 8B/14B, matched source/native controls, adapter-cost report |
| RepSpec | ICLR 2026 | accepted length, training/inference parameter accounting, matched EAGLE comparisons | position-wise acceptance, frozen/trainable parameter accounting, and EAGLE-3 second-family evaluation |
| SPEED-Bench | ICML 2026 accepted | domain diversity, context/concurrency sweeps, production-engine throughput, separation of qualitative and throughput workloads | current task breadth and strict batch-one scope; production claim withheld pending engine integration |
| Hierarchical Speculative Decoding | ICLR 2026 oral | separate formal distributional correctness from empirical speed and quality; vary model/task regime | target-authority proposition, conformance tests, paired official scores |
| GRIFFIN | NeurIPS 2025 | diagnose alignment degradation across draft positions rather than reporting only a mean | acceptance-survival curve for every proposed position |
| MoESD | NeurIPS 2025 | acceptance alone is insufficient; validate a systems performance model | measured CUDA-region decomposition and Amdahl prediction/reconstruction |
| Medusa | ICML 2024 | distinguish frozen-backbone adaptation from jointly trained higher-cost systems | frozen target/source/proposer declaration and trainable-parameter accounting |
| Heterogeneous-vocabulary speculative decoding | ICML 2025 | prove exactness under an adapter, then report wall-clock behavior across tasks | narrow compatible-family scope and target-only token authority |
| Distributed Speculative Inference | ICLR 2025 | distinguish algorithmic losslessness from observed finite-precision equality | formal exact-arithmetic statement plus byte-mismatch audits |
| Cross-Model KV Cache Transfer | arXiv 2026 preprint | current linear cross-model representation-transfer neighbor; expose adaptation size/time and accuracy retention | novelty boundary and measured 52--66M adapter/time/data table |
| Verifier Skipping | arXiv 2026 preprint | verifier work is a separate bottleneck; skipping it is a lossy axis | RelaySpec explicitly keeps every verifier call and claims only source-trunk elimination |
| SpecSA | arXiv 2026 preprint | verification kernels can dominate in long-context sparse-attention regimes | dense-SDPA scope; no cross-kernel extrapolation |

## Result acceptance logic

The literature motivates what to measure, while the decision rule comes from
the intervention itself:

1. The full target must remain the sole token authority and cache-conformance
   tests must pass.
2. The paired end-to-end speed confidence interval must lie above one.
3. Acceptance retention must exceed the break-even value derived from the
   measured source, relay, verifier, proposer, and residual time fractions.
4. Official paired quality and all finite-precision sequence disagreements are
   reported; exact text equality is not substituted for the proof.
5. A conventional full benchmark may be reported for literature comparability,
   but only the predeclared 468-prompt complement is called independent of the
   32-prompt design selection.

Primary-source URLs, checked versions, and publication status are recorded in
`docs/research/relayspec-source-log.md`; citation records are in
`references.bib`.
