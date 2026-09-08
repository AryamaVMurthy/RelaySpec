# Writer assessment and integration draft

Reviewed 2026-09-08 against the current manuscript, generated tables, family-scale reports, the 128-question protocol/status, the full 16,384-generated-record archive reproduction, and the cross-tokenizer generation implementation. This is a proposed main-text rewrite; it does not modify the manuscript. The main author should retain the bibliography verification work in the citation audit.

## The strongest coherent story

**A trained feature-conditioned drafter is more reusable than its original model-specific interface suggests. Replacing that interface with linear calibration preserves useful proposal behavior while removing a source transformer from generation. The evidence supports both a cheap, small-data operating point and a richer calibration recipe that reaches native performance.** The result is valuable because of what stays frozen and what disappears at inference, not because linear regression itself is new.

Three contribution points, in this order:

1. **A deployable retargeting method.** Isolate and replace the conditioning boundary of two released drafter families; keep the prediction network and target frozen; obtain useful acceleration, native throughput retention, and source-transformer memory savings. Non-Qwen and cross-tokenizer transfers broaden the tested deployment setting beyond increasing target size inside Qwen.
2. **An experimentally established calibration tradeoff.** Separate unique records from fitting work; examine data from 16 to 32,768 examples, matched linear/MLP capacity, regularization, longer fitting and prediction-based adaptation. Freeze selected checkpoints and confirm the small-data result on 256 GSM8K questions. Separately, retain the successful 16,384-generated-record layer-plus-context recipe: it reaches native parity with a repeated 2.6% advantage under a matched batch-invariant runtime. This supports a cheap default and a higher-effort operating point; it does not isolate a loss or data-size effect.
3. **A useful interface-level insight with limits.** Better feature regression is not a sufficient proxy for decoding speed. Extra calibration helps the Qwen-to-Llama transfer but not the selected Llama-to-Llama comparison. Feature selection also compresses the input of a native frozen drafter, demonstrating that the interface analysis has a use beyond retargeting; this deserves a short main-text paragraph with detailed native experiments in the appendix.

Do not claim first adapter-only tuning, first reusable drafter, universal linear optimality, a universal 512-record threshold, or superiority over all competing algorithms. TriSpec, PARD, and SD² make those formulations especially vulnerable. The public-runtime result is useful context: it is not the experiment that identifies RelaySpec's mechanism. The matched native/source/AR controls and matched mapper/adaptation sweeps do that.

**Important correction to the earlier reading:** the full archive reproduction is positive evidence for the alternate layer-plus-context recipe. Its weaker recent Llama/cross-family development results do not justify a global "ZIP did not work" conclusion. The positive Qwen transfer has the same total 52.43M trainable weights as the original dense map, but uses five separately supervised maps folded through the frozen fusion matrix. The simplicity claim can remain about a linear, frozen-drafter deployment; the training objective has more structure.

Suggested title: **RelaySpec: Reusing Frozen Speculative Drafters through Learned Linear Interfaces**. The existing small-data title is also defensible, but the proposed title foregrounds the operation whose value all experiments explain and accommodates the 8k cross-family result.

## What the newest evidence actually establishes

The 128-question protocol freezes configurations and uses MATH500 shuffled with seed 1729, indices 40:168. These exclude the preceding 40 questions in the recent family campaign. They are **not certified globally unseen in the wider project**. Use "reserved from the family tuning campaign," not unqualified "held out" or "fresh test set."

All methods obey a 2,048-output-token cap and normal EOS stopping. Target/mapper use FP32 with TF32 disabled; frozen drafter, source embedding and source head use BF16. This reference is separate from the primary BF16 Qwen benchmark. Cross-family uses tokenizer bridging and shared-token-end feature alignment, so the old manuscript's global shared-vocabulary restriction is no longer true.

| Reused drafter → target | AR request TPS | Old map request TPS | Selected map request TPS | Selected / AR | Selected / old | Direct full-token matches |
|---|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B → Llama-3.2-3B | 45.731 | 114.933 | 115.445 | 2.524 | 1.00446 | 128/128 for each map |
| Qwen3-4B → Llama-3.1-8B | 21.610 | 51.973 | 59.413 | 2.749 | 1.14315 | 128/128 for each map |

Use **request TPS** in the main paper, consistently with its defined primary metric. Decode TPS are 45.894/116.002/116.521 for Llama and 21.664/52.294/59.834 for cross-family; these belong in an appendix or an explicitly labeled extra column. Mixing 2.54×/2.76× decode ratios into an end-to-end headline is unnecessary.

The new cross-family mapper has 8,192 fitting records; the selected checkpoint is epoch 3 **of a 24-epoch cosine schedule**. Its 128-question request throughput gain is 14.3%. This is an expanded calibration/selection comparison, not a pure causal data-size ablation against the historical mapper: their data recipes and fitting budgets differ. The new Llama candidate is a 16,384-record epoch-6 checkpoint from a 12-epoch fit and gains only 0.45%. Retain the old, cheaper Llama mapper as the practical recommendation unless the new confidence interval provides contrary evidence. Both candidates should remain in the table to show the negative result.

The 128-question evaluation emitted 62,608 target tokens per Llama arm and 92,610 per cross-family arm. Eight and 24 requests respectively hit the cap. Direct equality of saved arrays gives zero sequence disagreements across 512 mapper/reference comparisons; this is not a mathematical-answer score or a proof for untested floating-point executions. The abstract should mention per-pair 128/128 rather than inflate the sample count to 512 independent questions.

## Nine-page allocation and evidence selection

Target approximately nine **main-text** pages in the official style, with references/appendix following the main-text marker. Do not reduce font sizes or hide protocol caveats to meet the limit.

| Main-text component | Page budget | What it must accomplish |
|---|---:|---|
| Abstract and introduction | 1.2 | Concrete deployment problem, frozen parts, three contributions, one compact headline visual |
| Related work and positioning | 0.9 | Closest approaches first; distinguish inherited-feature reuse from target-independent drafting and drafter adaptation |
| Preliminaries and method | 1.4 | Source/target definitions, exact interface/loss, inference data flow, cross-tokenizer extension and numerical reference |
| Setup and primary acceleration | 1.5 | Data exposure, timing, AR/native/source roles; main Qwen results and task quality |
| Calibration and interface findings | 1.7 | Data scaling + frozen confirmation; throughput/capacity result; brief native compression and adaptation comparison |
| Rich calibration and transfer across model families | 1.3 | Generated-rollout/native-parity result; new128 table, precision contract, exact matches, cross positive/Llama negative result |
| Public alternatives, discussion and conclusion | 1.0 | Public-runtime table with its confound explicit; operating regime, limitations, closing takeaway |
| **Total** | **9.0** | |

Keep in the main paper:

- Core Qwen MATH results and paired task-quality evidence; native retention and source removal must be visible next to the AR headline.
- A readable data-scaling curve covering the requested 16–32,768 range; put the 256-question frozen confirmation immediately after it.
- A **throughput-oriented capacity comparison**, or a compact table of dense/factored-linear/MLP results. The current validation-loss-only figure does not directly demonstrate the asserted decoding advantage. If reusing it, pair it visibly with throughput values, not distant prose.
- The new 128-question transfer table and finite-precision distinction. This answers the obvious reviewer objection that the mechanism might only work within Qwen.
- The generated-rollout recipe's native-parity result, with both 128-prompt repeats visible in a small table or paired paragraph. It is the strongest current test of whether the reused drafter can match a native one. Do not pool its AR-relative ratio with the historical BF16 main suite: the runtime and questions differ.
- A short, explicit matched adaptation comparison and public-runtime baseline table, with different purposes clearly stated.
- Native-interface compression as a short measured finding, with the code-capacity exception in the same paragraph.

Move detail to the appendix while retaining a one-sentence main-text finding:

- Full regularization grids and extended fitting trajectories.
- 14B 16/32-question capacity screens, learning-rate follow-ups, composition details, difficulty/length subgroup tables.
- All preliminary 8/16-question family results now superseded in scope by the 128-question evaluation, including the speculative initial BF16 results.
- Detailed ZIP reproduction provenance, competing layer-plus-context loss histories, per-checkpoint block search. Keep the positive full generated-data reproduction in main; say only that the alternate loss did not displace the selected dense map in the recent Llama/cross-family development sweep. No 128-question comparison of those two losses was performed in that campaign.
- All tokenizer-bridge implementation cases, precision traces, normalization-epsilon audit, fitting schedule manifests, request-level hashes and token-array checks.

A nine-page draft cannot give every experiment a full subsection. The experiments should answer a small number of questions, not reproduce the chronology of the project.

## Draft abstract

Feature-conditioned speculative drafters are trained around the hidden states of a particular language model. RelaySpec reuses these drafters with a new target by learning a linear conditioning interface while keeping the prediction network frozen and removing the source transformer from generation. Across DFlash and EAGLE-3 retargeted from Qwen3-4B to 8B and 14B, the original fits achieve 2.35–5.11 times autoregressive throughput on MATH-500. Frozen-checkpoint tests show that 512 calibration examples retain 97.5% and 98.9% of the throughput obtained with 2,048 examples on 256 GSM8K questions. A richer, generated-rollout recipe reaches native DFlash throughput, exceeding it by 2.6% in two matched 128-prompt runs. Controlled capacity, optimization and adaptation studies support a low-cost dense interface while showing that lower regression error does not consistently accelerate decoding. Separate Llama and cross-tokenizer Qwen-to-Llama transfers reach 2.52 and 2.75 times matched FP32-target AR throughput, with exact token agreement on all 128 requests per transfer. Expanded calibration improves cross-family throughput by 14.3% but leaves Llama-to-Llama throughput essentially unchanged. These findings establish practical operating points for inherited drafters and characterize the data, capacity and numerical limits of interface-only adaptation.

This is approximately 200 words. Memory savings and native-interface compression remain in the body without crowding the abstract. If the main author prefers fewer number groups, keep generated/native parity and non-Qwen breadth; drop the 14.3% number from the abstract and retain it in its table.

## Draft introduction

Feature-conditioned speculative decoding invests substantial training effort in a drafter that predicts several tokens from a language model's internal states. A deployment may later need a different target model: a larger model, a smaller model, or a model from another family. The inherited drafter can still make proposals, but the new target does not supply the interface on which it was trained. Keeping the original source model supplies compatible features at the cost of another transformer in the generation loop. Training a new drafter introduces a different adaptation cost. This paper asks how much of an inherited drafter can be reused by changing only its conditioning interface.

RelaySpec learns a linear map from the new target's hidden states to the context consumed by a frozen drafter. During calibration, the source and target process shared text and provide paired feature supervision. During generation, verification already computes the target features needed for the next draft. The learned map turns these states into the drafter's input, allowing the source transformer to be removed. The target, drafter and inherited token embedding and vocabulary projection remain fixed. The target verifier continues to determine which tokens are committed.

The useful claim is therefore operational: preserve a trained drafter's proposal behavior while replacing the expensive path that supplies its features. Linear alignment is not intrinsically new, nor is adapter-only tuning. Its value here depends on whether the resulting proposals remain good enough to compensate for the changed context and mapping overhead. We measure this with matched AR, native-drafter and source-reuse controls. On the main Qwen MATH-500 comparisons, RelaySpec reaches 2.35–5.11 times AR throughput and retains 89.1–90.3% of the evaluated native drafters' throughput. Removing source layers separately saves 7.63–7.80 GiB in EAGLE-3. Code workloads also reveal cases where source reuse remains faster, making the deployment tradeoff explicit.

Our contributions are threefold. **First, we make inherited feature-conditioned drafters reusable through a calibrated input interface.** We implement the mechanism in DFlash and EAGLE-3 and extend DFlash to Llama targets and cross-tokenizer Qwen-to-Llama reuse. **Second, we establish useful calibration operating points rather than assuming that a simple map is sufficient.** Frozen 512-example maps retain 97.5% and 98.9% of their 2,048-example counterparts' speed on 256 GSM8K questions, while a separate richer generated-rollout recipe reaches native DFlash performance in two 128-prompt runs. Matched linear/MLP capacity, regularization, longer fitting and token-prediction adaptation studies support direct context regression as a strong low-cost default. **Third, we characterize which interface improvements matter for generation.** Better regression and more optimization need not yield faster decoding; expanded calibration improves cross-family throughput by 14.3% on a separate 128-question evaluation while leaving Llama-to-Llama throughput essentially unchanged. Native-interface compression provides a second application of the same feature analysis.

Together, these experiments identify a useful unit of adaptation: the context supplied to a trained drafter. The contribution is a reusable inference path and an empirical account of its data, capacity and numerical limits, rather than a claim that linear maps or drafter reuse originate here.

Integration notes: add the existing DFlash/EAGLE citations to paragraph 1 and nearest-prior-work citations to paragraph 3 or the related-work section. The final paragraph can be cut if the page is full; it restates the organizing idea, not new evidence. Move tight accuracy noninferiority limits into the confirmation paragraph and limitations if they interrupt the contribution list, but do not omit them from the main text.

## Draft richer-calibration subsection

### Can richer calibration close the native-drafter gap?

The small-data studies hold a lightweight context-regression recipe fixed. We separately evaluate a more expensive linear calibration recipe using 16,384 newly generated NuminaMath rollouts and equal-weight layer and fused-context alignment. Five bias-free maps are folded through the frozen source fusion matrix for deployment; the drafter remains frozen. Three epochs use 7,911 updates and approximately 14.4 minutes of mapper optimization, following separately measured rollout generation and feature capture. This changes supervision, data and optimization together, so it is an operating-point comparison rather than a controlled attribution to one loss term.

On 128 archive evaluation prompts capped at 2,048 tokens, the mapped 4B drafter reaches 167.83 tokens/s against 163.57 for native 8B DFlash. Reversing GPU assignments gives 167.97 versus 163.67 tokens/s. The relay/native ratios are 1.0260 [1.0180, 1.0340] and 1.0263 [1.0182, 1.0342]. This separate batch-invariant BF16 runtime produces identical full token arrays for AR, native and mapped decoding on all 128 prompts; its AR baseline is 26.05 tokens/s and is not rerun in the second speculative comparison. The result supports native-level performance under this richer recipe, with a repeated 2.6% measured advantage. It does not replace the low-data cost comparison or establish that extra records alone caused the gain.

Keep archive-reproduction fidelity in the appendix: matching model/code/source-fusion hashes and evaluation outputs, but newly realized training rollouts and different final mapper hashes. Do not claim bit-for-bit training reproduction. The inference behavior was reproduced on this benchmark.

## Draft transfer subsection

### Does the interface transfer beyond Qwen?

We reuse a Llama-3.1-8B DFlash drafter with Llama-3.2-3B and a Qwen3-4B drafter with Llama-3.1-8B. The second transfer aligns fitting features at shared token-end positions and bridges proposals between tokenizers at inference; the target still verifies target-vocabulary tokens. We compare the previous mapper with a dense mapper selected from 4,096/8,192/16,384-record and multi-epoch development sweeps. Block lengths are frozen at 10 and 16. The 128 MATH questions exclude the preceding 40 questions in this family campaign; they are not claimed globally unexposed in the broader project.

The selected maps reach 115.44 and 59.41 end-to-end tokens/s, or 2.52 and 2.75 times their matched AR baselines (Table X). Expanded calibration improves cross-family throughput by 14.3%; the Llama-to-Llama gain is only 0.45%, supporting retention of its cheaper previous fit. The historical and new fitting recipes differ, so this comparison establishes an improved selected transfer configuration rather than an isolated data-size effect. An alternative layer-plus-context alignment loss and longer fitting did not displace the selected dense recipe in development (Appendix X).

Each old and selected mapper exactly matches all 128 AR token sequences through EOS or the common 2,048-token cap. These measurements use FP32 target/mapper execution with TF32 disabled and a BF16 frozen drafter. They establish agreement with this matched FP32 reference, not with the BF16 stream used in the main Qwen suite. Earlier BF16 disagreements motivated this numerical control. Exact sequence agreement is a finite-output validation, separate from mathematical correctness; 8 and 24 requests respectively reach the cap.

## Draft conclusion and limitations

RelaySpec reuses a trained speculative drafter by calibrating its input context and removing the source transformer from generation. Its value is retaining learned proposal behavior while replacing the dependency tying that behavior to one source model. The evidence supports both a small-data direct-regression recipe and a richer linear calibration recipe that reaches native throughput. Capacity, optimization and adaptation studies characterize when extra fitting helps, while the Llama transfers establish reuse beyond the primary Qwen setting. The result is a practical interface-level alternative to retraining a drafter, with measured speed and memory tradeoffs.

The evidence spans two drafter families, Qwen and Llama targets, and a cross-tokenizer transfer, but remains limited to one request per worker and the evaluated model pairs. Development exposure, related math templates and limited fitting seeds constrain generalization. The public-baseline comparison uses separate runtime controls and does not isolate an algorithm ranking. BF16 output differences and exact FP32-target matches must be read under their respective numerical references. Frozen small-data confirmation supports throughput retention, but not a tight accuracy noninferiority claim. These boundaries define the current operating regime rather than a universal replacement for training a target-specific drafter.

## Contradictions and wording to fix before completion

1. Abstract/introduction/method/conclusion repeatedly equate retargeting with a **larger** target. General statement must be "new target"; retain "larger" only for the named Qwen 4B→8B/14B experiments.
2. Related-work statement "RelaySpec studies offline fitting with a shared vocabulary" is obsolete globally. Change to a primary shared-vocabulary study and an explicitly described cross-tokenizer extension; retain heterogeneous-verification citations without implying a new generic verifier.
3. Conclusion says only shared-vocabulary Qwen models. Replace with actual expanded scope and its restricted pair count.
4. Global experiment setup and appendix protocol table currently say BF16, Qwen-only, block16. Label them **primary Qwen suite** and add a separate precise family-transfer protocol.
5. The equations/record-loss description describes the original fit. The cached-feature family campaign samples positions and has distinct normalization, schedule and data caps. Do not present every reported fit as the same 4,096×192-token/1,024-update training procedure.
6. Native-throughput and memory claims come from specific suites. Do not imply those controls were measured for the new Llama/cross transfers.
7. "More data/longer fitting do not help" must become "do not consistently help": the new cross-family gain is positive evidence against a universal saturation statement.
8. "Feature regression wins every tested adaptation budget" is limited to the measured 512-record warm-budget CE/LoRA development comparison, not every loss/objective ever run or unbounded tuning.
9. Previous family recommendations/reports contain superseded 16-question numbers and some initial full-FP32 recommendations. Treat them as historical appendices; the new 128-question mixed-precision contract is the current result.
10. Inspect rendered nine-page main text after integration. Large fixed-position figures currently interrupt the conceptual sequence; resizing or relocating redundant figures should precede prose cuts that remove essential protocol distinctions.
11. Add the full 16,384-generated-record layer-plus-context result. It contradicts a global claim that direct context regression was always the best recipe. Keep the true narrower claim: dense direct context fitting led matched 8B capacity/budget comparisons, and richer structured linear calibration separately reaches native throughput.
12. The archive recipe has a distinct target batch-invariant BF16 runtime and source/drafter attachment. Its 6.44× AR ratio must not silently extend the historical 2.35–5.11× same-runtime Qwen headline. Present the suite identity next to the number, or headline its native parity instead.
