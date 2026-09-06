# RelaySpec competitor priorities

Primary papers and official repositories checked on 6 September 2026. This assessment supersedes the implication that the PARD/SD-square table covers all closest competitors. Rankings below are research judgments, not measured performance rankings. No new experiment was launched.

| Priority and purpose | Candidate | Relationship and current evidence |
|---|---|---|
| Essential matched deployment controls | Native DFlash/EAGLE-3, optimized source reuse, AR | Already measured. They test the cost of reusing the inherited drafter instead of deploying a target-specific checkpoint or retaining the source transformer. Native checkpoints are references, not matched training-cost experiments. |
| Highest-priority new external performance run | PARD-2, with original PARD retained | Target-dependent and target-independent modes address the performance/reuse tradeoff. Official repository lists Qwen3-8B and 14B checkpoints and inference code. Original PARD was evaluated here; PARD-2 was not. |
| Highest-priority novelty comparison | TriSpec adapter-only variant | Its Section 3.2 explicitly maps proxy features into an EAGLE-family drafter and permits a frozen drafter with only the adapter trained. This is direct prior art for feature adapters into frozen speculative drafters. The full system instead reduces target verification through a confidence-gated proxy, with an accuracy/latency tradeoff. RelaySpec removes source conditioning at inference and retains target verification. No official implementation was located in this search; that does not establish its absence. Not evaluated here. |
| Closest efficient-adaptation problem comparison | Efficient Draft Adaptation (EDA) | Adapts after domain fine-tuning of a target, using shared/private draft components, regenerated data and sample selection. Official code documents Qwen2.5 domain targets and staged training. Different-sized Qwen3 transfer needs an explicit compatible protocol rather than relabeling our LoRA baseline as EDA. Not evaluated here. |
| Secondary mechanistic competitor | SD-square | Uses target hidden states to steer a pretrained standalone drafter. Main paper trains the steering and drafter together and reports a frozen-drafter ablation. Our experiment covers only that frozen option under the recorded small-data recipe and public runtime. It cannot establish superiority over full SD-square. |
| Foundational adaptation comparison | DistillSpec | Knowledge-distillation alignment of a drafter to a target. Cite as a direct alternative to feature regression. Our hard-label CE/LoRA screens are not a faithful DistillSpec reproduction. |
| Broader scope comparison | OmniDraft | Cross-vocabulary online adaptation is relevant to universal reuse but differs from our offline shared-vocabulary problem. Already discussed, not evaluated here. |

## What should be run next

1. Evaluate the released PARD-2 Qwen3-8B checkpoint in both supported modes, retaining original PARD and AR. Use the same questions, target revision, prompt, thinking mode, cap and decoding policy as RelaySpec. Record engine, precision, attention, compilation and timing boundaries. Prefer a common runtime; otherwise retain separate AR controls and identify this as a deployment comparison. Published speeds must not be mixed with our measurements.
2. For an additional training experiment, prioritize a faithful adaptation comparison informed by EDA, with explicit model compatibility and all preprocessing/training costs. A port is not automatically a reproduction. Establish feasibility before expensive fitting.
3. Address TriSpec in novelty positioning immediately. If reproducing its adapter component, label a component adaptation separately from the complete proxy-verification system. Do not place relaxed verification and target-verified methods in one unconditional lossless speed ranking.

SD-square expansion is lower priority than these items. Medusa, tree scheduling, serving engines and unrelated new drafter architectures are broader context unless the paper's scope expands to universal speed leadership.

## Required positioning correction

PARD-2 and EDA already have bibliography entries but are absent from the current cited main-text comparison. TriSpec was missing from this positioning audit. The revised novelty claim must not imply that learning an adapter into a frozen speculative drafter is itself unprecedented. The supported distinction is source-removing retargeting of inherited drafters, target-controlled verification, and the demonstrated small-data/capacity/deployment behavior across the two tested families. The current table is a table of completed evaluations, not an exhaustive table of the closest prior work.

## Primary sources and artifact links

- PARD (ICLR 2026): https://arxiv.org/abs/2504.18583
- PARD-2 (2026 preprint; no venue confirmed here): https://arxiv.org/abs/2605.08632
- Official PARD/PARD-2 implementation: https://github.com/AMD-AGI/PARD
- Opened Qwen3-8B checkpoint: https://huggingface.co/amd/PARD2-Qwen3-8B . The repository also lists `amd/PARD2-Qwen3-14B`; its model page returned an error during this check.
- TriSpec (2026 preprint; OpenReview submission found, acceptance unconfirmed): https://arxiv.org/html/2601.23180 , Sections 3.2 and 4.4.
- EDA (2026 preprint): https://arxiv.org/html/2603.09527 , Section 4. Official code: https://github.com/Lyn-Lucy/Efficient-Draft-Adaptation . The README's title/year differ from the linked paper, so use the paper's citation metadata.
- SD-square (AAAI 2026): https://ojs.aaai.org/index.php/AAAI/article/view/40255 . Training and frozen ablation: https://arxiv.org/html/2511.09844 , Sections 3 and 4.1.
- DistillSpec (ICLR 2024): https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html
- OmniDraft (NeurIPS 2025): https://papers.nips.cc/paper_files/paper/2025/file/3c2fe1417eed1c6ff9acf169617981ea-Paper-Conference.pdf
