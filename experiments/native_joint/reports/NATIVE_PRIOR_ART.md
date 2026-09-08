# Prior art affecting the standalone native study

Checked primary paper abstracts and official repositories on2026-09-08. This is a positioning map, not a benchmark ranking. Published headline speedups use other hardware and protocols and are not comparable to this study's local TPS.

| Work | Relevant overlap | Local status |
|---|---|---|
| [DDTree](https://arxiv.org/abs/2604.12989), Ringel and Romano, April2026 | Best-first trees from DFlash parallel marginals, with packed target verification. | MIT heap builder adapted from official commitc96427a185677bf4133ed865dd1626a5041aef9b; measured in the shared pinned runtime. This is existing-method evidence. |
| [DART](https://arxiv.org/abs/2601.19278), Liu etal., January2026 | Parallel token predictions plus n-gram continuity in tree construction. | Not reproduced. Our prompt-history augmentation is a small in-context heuristic, not DART reproduction or an established novelty. |
| [JetSpec](https://arxiv.org/abs/2606.18394), Hu etal., June2026 | A causal parallel draft head creates branch-conditioned trees from frozen target features. | Not reproduced. General causal parallel-tree training cannot be claimed new here. |
| [Weaver / Trees from Marginals](https://arxiv.org/abs/2607.06763), Oda etal., July2026 | A lightweight autoregressive adapter restores dependencies over a factorized drafter's top-K candidates. | Not reproduced. Relevant before proposing a learned conditional reranker. |
| [PCTree](https://arxiv.org/abs/2608.02123), Li etal., August2026 | Uses DSpark's existing Markov head to score parent-specific alternative continuations. | Not reproduced. Relevant to any extension from chain correction to tree correction. |
| [DARTree](https://arxiv.org/abs/2608.13524), Li etal., August2026 | Extends pretrained autoregressive correction from diffusion draft chains to trees. | Not reproduced. Relevant to branch-wise refinement proposals. |

Inference from these sources: faster native speculative decoding is an active and crowded area. The general ideas of parallel trees, lightweight conditional correction and continuity-aware selection have direct prior art. A new contribution needs a specific demonstrated improvement over appropriate existing methods, or a distinct capability with supporting evidence. Our joint compression, midpoint injection, local top-five loss and fixed leaf policies have not yet improved on full DDTree in the completed direct comparisons. Keep this separate from RelaySpec's drafter-transfer contribution; this file does not change the RelaySpec manuscript.


JetSpec's official repository documents both an HF reference path and an optimized engine with paged attention/CUDA graphs. Its declared Transformers dependency is>=4.51,<5 (validated4.57.1), whereas our shared Turing environment is5.3.0. Any reproduction should use an isolated compatible environment and explicitly distinguish algorithm and runtime comparisons. No shared environment has been modified. Source: https://github.com/hao-ai-lab/JetSpec/blob/master/pyproject.toml and official README, inspected2026-09-08.


JetSpec source inspection is pinned to commit2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f, cloned read-only under `/tmp/native-jetspec-reference`. In `bench/reference/benchmark.py`, the optional linear “DFlash” comparison receives the **same selected drafter object** as the JetSpec tree path. With a JetSpec checkpoint selected, that is a same-head linear-vs-tree comparison, not a separately loaded original released DFlash checkpoint. A direct reproduction must explicitly retain our original released-DFlash control. No JetSpec model has been downloaded or benchmarked in this study yet.
