# Method comparison verification

The main paper now separates learned operations (Table 1) from deployment capabilities (Table 2). The use case is a released feature-conditioned drafter, a new target within a shared vocabulary, fixed drafter and target weights, and removal of source-transformer conditioning. Checks describe the published procedure. Dashes do not claim that an extension is impossible. No head-to-head speed superiority over PARD, SD² or OmniDraft is inferred.

| Method | Primary evidence | Verified distinction |
| --- | --- | --- |
| DFlash | https://arxiv.org/abs/2602.06036 | Trains a block diffusion drafter conditioned on target features. Native training does not reuse the old feature-conditioned head unchanged. |
| EAGLE-3 | https://arxiv.org/abs/2503.01840 | Direct token prediction with multi-layer target features and training-time test. Its removal of feature prediction must not be confused with removal of feature conditioning. |
| PARD | https://arxiv.org/html/2504.18583v4 | Adapts an autoregressive model to parallel drafting with masked prediction and conditional token dropping. Target-independent reuse is different from retargeting a released dependent head. |
| SD² | https://arxiv.org/html/2511.09844v1 | Sections 3 and 4.1 describe target-feature steering into drafter MLPs, KL alignment, default drafter fine-tuning, and frozen-drafter ablations. Frozen variants explicitly receive a daggered check. A linear map and freezing alone are not RelaySpec novelty. |
| OmniDraft | https://proceedings.neurips.cc/paper_files/paper/2025/hash/3c2fe1417eed1c6ff9acf169617981ea-Abstract-Conference.html | Cross-vocabulary online adaptation uses an n-gram cache and hybrid distillation fine-tuning. More relevant to reuse across targets than generic drafter-training improvements. |

RepSpec and Draft-OPD remain cited in the additional related-work appendix. Their main contribution concerns training a drafter, so they are secondary to the reuse and steering methods in the main comparison. The source-reuse control is included in the capability table because it already meets the frozen-weight requirements but retains source-transformer computation. Independent drafters receive credit for not requiring a source transformer.

The search also inspected Halfway Speculative Decoding (https://sergiu-nistor.com/assets/publications/Halfway_Speculative_Decoding.pdf). It jointly trains an EAGLE-style head and target LoRA using an acceptance objective. It is not a frozen source-conditioned head retargeting method, so an additional preprint citation was not needed for this comparison. This search is not a proof of priority over every published method.

RelaySpec's claim is the source-context fitting and source-removal procedure, integrated with the released DFlash and EAGLE-3 normalization conventions and evaluated across two target sizes. Its observed advantages are reported against paired AR, native-drafter and source-reuse controls. Existing experimental records were retained without alteration.

The revised paper has nine main pages, 18 total pages, eight figures, 17 tables and 42 references. Broader related work is Appendix G. Main figures use 90 percent linewidth, with official style files unchanged.
