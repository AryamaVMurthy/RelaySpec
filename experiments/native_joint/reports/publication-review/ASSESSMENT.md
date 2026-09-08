# Accuracy, novelty, and a separate ICLR paper

Assessment date: 2026-09-08. Scope: the standalone native study and its frozen 128-request confirmation, not RelaySpec transfer. This is a targeted literature and code assessment, not an exhaustive novelty search or an acceptance prediction.

## Accuracy: the theoretical answer is yes, subject to exact verification

For greedy decoding, let the target's next token at prefix h be g(h) = argmax_v p(v | h), with a fixed tie rule. A valid speculative verifier commits only draft tokens matching g at the corresponding true prefix. At the first mismatch it emits g instead. A tree may provide several possible continuations, but its committed path must obey the same rule. Induction on the committed prefix gives exactly the autoregressive sequence, hence exactly the same deterministic task score. Poor proposals reduce accepted progress; they do not introduce an inherent accuracy tradeoff.

This assumes identical target conditionals, causal/tree masks, positions, cache state, tokenization, logits processing, tie rules and stopping rules. The stochastic guarantee of a correct exact sampler is equality of output distributions and therefore expected task scores; individual sampled answers and finite-sample accuracies need not agree, even when seeds have the same integer value. Our current native runner implements greedy decoding only. The foundational guarantee is established in [Leviathan, Kalman and Matias, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html).

The code in `radical_decode.py` uses ancestor visibility, depth-based positions, cumulative prefix matches, target correction and selected-path cache compaction. These implement the intended greedy rule. Tests and inspection are evidence for the implementation; they are not a proof of bitwise equality to the separate AR execution path.

Our BF16 results do not meet that stronger execution-level requirement: full DDTree exactly matches AR on 42/128 requests, and original DFlash itself matches AR on 40/128. Original versus full DDTree scores are 28 versus 29 on GSM8K, 30 versus 28 on MATH, and 31 versus 30 on HumanEval, each out of 32. This does not show that speculation intrinsically sacrifices accuracy, nor does it justify claiming equal practical accuracy. Small differences are unresolved; absence of a significant difference would not prove equivalence.

[PyTorch's numerical documentation](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html) explains that mathematically equivalent batched and sliced computations need not be bitwise identical. A small perturbation can change the argmax near a tie and then alter the generated suffix. Earlier local diagnostics found tied BF16 logits in four selected fixed-tree divergences and recovered sequence equality through 512 tokens with FP32 parameters. That does not diagnose every final DDTree divergence. FP32 itself is not a universal guarantee of layout-invariant arithmetic.

Before an exactness claim, capture every first divergence at the same true prefix, compare masks/positions and fresh versus reused caches, and replay with a defined reference execution schedule. Distinguish a correct mathematical algorithm, reproducibility of repeated execution, cross-layout token agreement, and task quality. A stability condition such as top-two logit margin > 2 epsilon only certifies an argmax when epsilon is a valid bound on each logit's execution error; an observed margin threshold alone is not a certificate.

## Novelty: the winning method is prior art

Our fastest arm uses the released full drafter plus an adapted DDTree heap builder. `ddtree_baseline.py` explicitly credits upstream commit c96427a185677bf4133ed865dd1626a5041aef9b and its MIT license. The measured 17.6% improvement is useful reproduction evidence, not our algorithmic invention. The compact joint-training arm is 6.1% slower than original DFlash; compact plus tree is 9.7% faster than original but 6.7% slower than full DDTree. See [the completed decision](../DECISION.md).

| Proposed claim or direction | Closest verified prior art | Consequence |
|---|---|---|
| Trees from one-pass diffusion marginals | [DDTree](https://arxiv.org/abs/2604.12989) | The principal positive result is an existing-method baseline. |
| Select tree size using latency and accepted progress | [CaDDTree](https://arxiv.org/abs/2606.01813) | Generic cost-aware node budgeting is already addressed. Our affine-cost screen is not a CaDDTree reproduction. |
| Train using survival through sequential verification | [Verification-Aware Training](https://arxiv.org/abs/2608.30135) | A verification head and rejection-dependent training weights are already proposed for DFlash and EAGLE-3. |
| Learn to align a small draft model for acceptance | [DistillSpec, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/8766fbc68e1ed1cdef712ce273e0a363-Paper-Conference.pdf) | Distillation for acceptance is established; a new interface configuration alone is a weak novelty claim. |
| Branch-conditioned parallel proposals or lightweight correction | [JetSpec](https://arxiv.org/abs/2606.18394), [Weaver](https://arxiv.org/abs/2607.06763), [PCTree](https://arxiv.org/abs/2608.02123), [DARTree](https://arxiv.org/abs/2608.13524) | The broad architectural direction has multiple direct predecessors. |
| Target-aware, prefix-conditioned tree selection | [TAPS](https://arxiv.org/abs/2606.00487) | Replacing independent marginal scores with target-aware prefix estimates is not an unoccupied idea. |
| Prefix confidence plus load-aware verification | [DSpark](https://arxiv.org/abs/2607.05147) | Deployment-dependent verification scheduling also has direct overlap. |
| Restore deterministic inference with verification and rollback | [LLM-42](https://arxiv.org/abs/2601.17768) | Numerical drift and replay-based determinism cannot by themselves be pitched as new. |

The 2026 arXiv papers above were checked as primary preprint versions; peer-reviewed publication status was not established for most of them. Public preprints still matter for novelty. Their reported headline speedups are not a valid ranking against our shared-runtime measurements. Only DDTree among these new tree/training baselines was actually measured locally. CaDDTree, VAT and TAPS were missing from our earlier positioning map and materially narrow the available claims.

Additional overlap checks found a public [SALF & TALF manuscript](https://openreview.net/pdf?id=3V559xWIWc) describing a tree-aware training loss, and [an empirical study on consumer hardware](https://arxiv.org/abs/2607.17283). The former PDF is labelled under review at ICLR 2026; its current decision was not verified. Merely proposing a tree-aware loss or reporting that some speculative settings slow down is therefore insufficient differentiation.

## ICLR assessment

My recommendation is **do not submit the current experiment package as a new fast native decoding method**. It has useful reproducibility work and negative evidence, but its strongest speed result belongs to prior art. The new compact candidate loses to the stronger local baseline, the confirmation covers one target family and one GPU type, and the execution-level accuracy discrepancy remains only partly diagnosed.

This is not a requirement to achieve state of the art at all costs. [ICLR 2027's reviewer guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) explicitly values significant new knowledge, including empirical and theoretical findings, without requiring state-of-the-art results. A carefully established explanation or capability could support a paper. The current evidence does not yet establish that broader contribution.

| Possible separate paper | Current support | Missing contribution | Assessment |
|---|---|---|---|
| A faster compact native drafter | Broad search; frozen negative result | A differentiated mechanism with meaningful gains over strong relevant controls | Not supported now |
| When does draft compression improve decoding? | Compact round time barely changes while accepted progress falls | Predictive, generalizable boundaries validated on unseen model/hardware/settings; evidence of both success and failure regimes | Most natural analysis direction from these results, still unproven |
| Practical exactness of speculative verification | Selected BF16/FP32 diagnoses and output disagreements | A specific new cause, diagnostic or low-overhead remedy beyond known shape sensitivity and LLM-42 | Worth investigating; high overlap risk |

For the analysis direction, start from throughput = committed tokens / elapsed time. Under a matched stationary per-round approximation, compression helps when its retained progress fraction exceeds its retained time fraction. This algebra is elementary and is not itself a new theorem. A contribution would be predicting those quantities from architecture and workload before expensive fitting, validating the prediction on held-out settings, and showing it changes deployment decisions. Our current numbers explain one failure; they do not establish a universal design law.

For a methods direction, a possible research question is whether compact drafting can optimize coverage of useful verified paths at the measured deployment cost. That question is not a novelty claim: CaDDTree, VAT, TAPS, JetSpec, DSpark and tree-aware objectives must be compared before implementation is pitched as new. Combining two existing techniques and renaming the combination would not by itself establish a substantial contribution.

## Concrete next evidence, in priority order

1. Finish the numerical correctness diagnosis using the already saved divergent outputs. Hold prefixes fixed; isolate execution shape, precision, cache and masking. Report any unresolved causes rather than labelling every mismatch rounding noise. The present confirmation set may support diagnosis, but it is consumed for final selection purposes.
2. Reproduce the most relevant missing controls: CaDDTree and VAT, with a VAT-plus-DDTree combination if compatible. For a branch-conditioned proposal, add a compatible JetSpec/Weaver/TAPS control. Keep original released checkpoints distinct from fine-tuned heads, and separate common-runtime algorithm comparisons from optimized-engine comparisons.
3. Test the compression explanation on at least one non-Qwen family and another target scale, then validate portability on another GPU architecture and a small, deliberate selection of batch/context regimes. Measure target/draft/packing costs, accepted progress, memory and task scores. These are proposed study sizes, not official ICLR requirements. A full Cartesian sweep is unnecessary initially.
4. Promote only a specific new finding or method. A method claiming a further 10% improvement must beat the strongest relevant reproduced baseline, not only original DFlash. At the old workload's 199.35 TPS baseline that would mean roughly 219.3 TPS, but a stronger control or new test set changes that numerical target. The 10% threshold is the user's research target, not an ICLR acceptance rule.
5. Freeze the selected claim and configuration, then use newly reserved questions, repeated paired timings, multiple fitting seeds when relevant, and adequate task scoring. Write the separate paper after these results establish its contribution. A technical report or suitable workshop study is a more defensible current output, with acceptance also unguaranteed.

No new GPU jobs or manuscript rewrites were launched for this assessment. The existing frozen inference source and results remain unchanged. Sources and verified metadata are recorded alongside this assessment.
