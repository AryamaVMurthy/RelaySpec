# What recent public reviews imply for RelaySpec

Read on 5 September 2026. This is a targeted study of **11 public discussions containing 42 official reviews**, including nine accepted papers and two rejected contrast cases. Six accepted papers are from ICLR 2025/2026, two from NeurIPS 2025, and one from ICML 2025. Decisions, reviews, and available meta-reviews were read on the rendered public OpenReview pages. PARD and OmniDraft rebuttals were also read in detail. This is not a representative statistical sample of conference decisions, an acceptance predictor, or independent reproduction of the authors' experiments.

The linked notes distinguish reviewer opinion from author claims. Meta-reviews sometimes speculate about eventual reviewer scores or contain inconsistent summaries; this study does not treat those predictions as actual score changes. Publication year differs from review date: several ICLR 2025 reviews were written in late 2024. No private reviews were accessed.

## Decision and feedback record

### 1. PARD — ICLR 2026 poster; four reviews

[Paper and discussion](https://openreview.net/forum?id=XbOyv7iVGL), [decision](https://openreview.net/forum?id=XbOyv7iVGL&noteId=Dz2cVj8AX9), [meta-review](https://openreview.net/forum?id=XbOyv7iVGL&noteId=ivFc2YHjZS).

Reviewers valued adapting an existing small model for parallel drafting and reusing it across targets. They challenged missing parallel-drafting comparisons, same-family confounding, acceptance as draft length increases, and fair chain-versus-tree evaluation. The authors supplied a cross-family initialization control and additional acceptance measurements. One reviewer still questioned whether restricting tree-capable baselines made the comparison fair. Acceptance did not resolve every objection.

**RelaySpec inference:** compare an identical search policy to isolate the relay's effect, and separately compare each baseline in its practical supported configuration. Include a genuinely different model family. PARD already supports target-independent reuse; do not describe all competing methods as requiring a fresh fit for every target. [Baseline fairness exchange](https://openreview.net/forum?id=XbOyv7iVGL&noteId=GuOnxVnpN4), [cross-family author experiment](https://openreview.net/forum?id=XbOyv7iVGL&noteId=cimFleQxRC).

### 2. RepSpec — ICLR 2026 poster; four reviews

[Paper and discussion](https://openreview.net/forum?id=bqEi97qzzz), [decision](https://openreview.net/forum?id=bqEi97qzzz&noteId=pvgd8ypU9v), [meta-review](https://openreview.net/forum?id=bqEi97qzzz&noteId=wmDwxgiPRn).

The method expands linear structures during training and merges them for inference. Reviewers saw practical value but questioned training overhead, modest improvements, limited novelty over established re-parameterization, and whether a nonlinear variant was meaningfully different from enlarging the drafter. The meta-review acknowledged additional larger-model results while retaining concerns about explanation and generalization.

**RelaySpec inference:** additional components need a measured benefit after their own cost. Compare equal-capacity linear and nonlinear maps, and report fitting cost separately from inference cost. A successful application of a familiar technique can matter, but its usefulness and mechanism need evidence. [Nonlinear-cost review](https://openreview.net/forum?id=bqEi97qzzz&noteId=aBiljL0mCA), [novelty review](https://openreview.net/forum?id=bqEi97qzzz&noteId=YxxxZxfl7k).

### 3. FastGRPO — ICLR 2026 poster; three reviews

[Paper and discussion](https://openreview.net/forum?id=zuGt6TYYtS), [decision](https://openreview.net/forum?id=zuGt6TYYtS&noteId=GHETgH3bsH), [meta-review](https://openreview.net/forum?id=zuGt6TYYtS&noteId=d1VEaWvJMz).

Reviewers appreciated a concrete bottleneck: generating training responses under changing concurrency and an evolving target. They asked for final learning quality, sampling diversity, stronger competing training systems, uncertainty, hardware calibration, and failure regimes. The meta-review contains inconsistent wording about whether broader-task evidence remained missing; that ambiguity is not resolved here.

**RelaySpec inference:** an online-training claim requires complete training runs, final quality, adaptation overhead, and concurrency-aware competitors. Static checkpoint transfer alone cannot support it. Keep reinforcement-learning rollout acceleration outside the core submission unless this substantially larger experiment is completed. [Quality concern](https://openreview.net/forum?id=zuGt6TYYtS&noteId=T7em7bZb29), [measurement concerns](https://openreview.net/forum?id=zuGt6TYYtS&noteId=jerFk8ZaRb).

### 4. OmniDraft — NeurIPS 2025 poster; four reviews

[Paper and discussion](https://openreview.net/forum?id=RALtozQipi), [decision](https://openreview.net/forum?id=RALtozQipi&noteId=oZYXaAi1wy).

Reviewers valued cross-vocabulary reuse and online adaptation, while asking about cache memory, long-stream stability, larger models, and actual device latency. Authors added cache and distribution-shift experiments. A reviewer explicitly raised their score from 3 to 4 while retaining deployment reservations. The authors' mobile results used component profiling and replay-based estimates, with cache lookup omitted; they were not a completed end-to-end device implementation.

**RelaySpec inference:** report the full runtime stack and distinguish measured deployment from projected performance. Larger drafters can accept more tokens yet run slower, so acceptance alone is not the objective. Keep adaptation instability and memory overhead visible. [Reviewer follow-up](https://openreview.net/forum?id=RALtozQipi&noteId=DlZOAgwyCq), [author measurement qualification](https://openreview.net/forum?id=RALtozQipi&noteId=NhVYnJnWIF).

### 5. EAGLE-3 — NeurIPS 2025 poster; four reviews

[Paper and discussion](https://openreview.net/forum?id=4exx1hUffq), [decision](https://openreview.net/forum?id=4exx1hUffq&noteId=txeau0sUpj).

The decision valued empirical gains, production-framework integration, ablations, and data-scaling analysis. It explicitly said architectural breadth, theory, smaller production/MoE gains, and initial reproducibility gaps reduced confidence for a higher presentation category. Reviewers requested exact layer choices, training steps, equal-data comparisons, training cost, and an explanation for different speedups between frameworks.

**RelaySpec inference:** publish exact tapped layers and update/data counts. Show an optimized-engine measurement without treating it as numerically interchangeable with the current PyTorch implementation. Explain why feature matching trains the connector even though EAGLE-3 itself removed a feature-prediction constraint from drafter training. [Configuration and engine review](https://openreview.net/forum?id=4exx1hUffq&noteId=hCT2ZTbwbz), [equal-data concern](https://openreview.net/forum?id=4exx1hUffq&noteId=U3vKvpfMol).

### 6. SWIFT — ICLR 2025 poster; four reviews

[Paper and discussion](https://openreview.net/forum?id=EKJhH5D5wA), [meta-review](https://openreview.net/forum?id=EKJhH5D5wA&noteId=qpgNc1xm38).

Reviewers valued deployment without an extra trained model. They asked whether speed included the layer-search phase, how many requests amortized that search, whether cheaply trained alternatives were competitive, and what happened when the input task changed. Several considered absolute speedups modest relative to trained drafters.

**RelaySpec inference:** make the adaptation budget and requests needed to recover its cost explicit. “No drafter retraining” is an operational advantage, not evidence of being the fastest method. If adding a controller, test task transitions and include its calibration overhead. [Cost and dynamic-setting review](https://openreview.net/forum?id=EKJhH5D5wA&noteId=vaHtgvrEtH).

### 7. Mixture of Attentions — ICLR 2025 poster; four reviews

[Paper and discussion](https://openreview.net/forum?id=Rz0kozh3LE), [meta-review](https://openreview.net/forum?id=Rz0kozh3LE&noteId=W1rl4EbbaV).

Reviewers appreciated diagnosing the information and training-context problems before introducing components. They requested a clearer overall diagram, explanations for design choices, component-removal ablations, latency/complexity, quality, and additional architectures. The meta-review identified increased complexity relative to a small improvement as a weakness.

**RelaySpec inference:** make each component answer an observed problem. The current two drafter integrations are valuable because they expose different input contracts. Explain the whole decoding cycle before equations. Do not assume an empirical claim in a review is correct merely because the paper was accepted. [Presentation and ablation review](https://openreview.net/forum?id=Rz0kozh3LE&noteId=VOaZSLmlfQ).

### 8. Faster Cascades via Speculative Decoding — ICLR 2025 oral; three reviews

[Paper and discussion](https://openreview.net/forum?id=vo9t20wsmd), [meta-review](https://openreview.net/forum?id=vo9t20wsmd&noteId=YBj6R2et11).

Reviewers valued a clearly defined combination of two mechanisms and a derived decision rule. Concerns included sampling fairness, baseline tuning, confusing method names, and insufficient intuition before theory. The meta-review describes additional sampling experiments and clarifications. This paper explicitly studies a cost–quality tradeoff; its objective is different from preserving an unchanged target distribution.

**RelaySpec inference:** define the deployment objective before reporting speed. Name every baseline and distinguish old methods from new variants. A simple derived rule can strengthen the explanation if its assumptions are explicit and its predictions are tested separately. Oral acceptance here does not imply that copying the same structure establishes novelty. [Fairness review](https://openreview.net/forum?id=vo9t20wsmd&noteId=vK9JV4A2i3), [clarity review](https://openreview.net/forum?id=vo9t20wsmd&noteId=CUFGzlg5TH).

### 9. Lossless decoding for heterogeneous vocabularies — ICML 2025 oral; four reviews

[Paper and discussion](https://openreview.net/forum?id=vQubr1uBUw), [decision](https://openreview.net/forum?id=vQubr1uBUw&noteId=gx2qp54uTb).

The decision valued removing a practical vocabulary restriction, correctness arguments, and integration into Transformers. Reviewers asked about tokenization overhead, standardized larger evaluations, uncertainty, and expensive theoretical variants. One explicitly flagged large generated-token-count differences under greedy decoding as a possible implementation defect, even though the paper was accepted.

**RelaySpec inference:** cross-tokenizer compatibility does not excuse unexplained target-output differences. Preserve target-token authority, test special tokens and context-dependent tokenization, and measure conversion costs. Distinguish theoretical feasibility from implemented speed. [Output-count concern](https://openreview.net/forum?id=vQubr1uBUw&noteId=7gg4X71RIv), [reproduction and overhead concern](https://openreview.net/forum?id=vQubr1uBUw&noteId=K4xA0fIMyc).

### 10. TriSpec — rejected at ICLR 2026; four reviews

[Paper and discussion](https://openreview.net/forum?id=yhhgkkiQe5), [meta-review](https://openreview.net/forum?id=yhhgkkiQe5&noteId=adw7O44s4g).

Reviewers found the verification-cost motivation clear. The meta-review identifies loss of the lossless guarantee, close prior work, baseline/truncation comparability, and deployment complexity as key blockers. Additional tasks, models, and uncertainty did not settle the central concerns.

**RelaySpec inference:** expanding the benchmark list cannot compensate for unresolved correctness or an unclear distinction from prior work. Keep the target as final verifier when exploring cheaper drafting. This contrast does not prove that all approximate accelerators are unpublishable: Faster Cascades was accepted under a different, explicitly stated objective.

### 11. The Disparate Impacts of Speculative Decoding — rejected at ICLR 2026; four reviews

[Paper and discussion](https://openreview.net/forum?id=2hOQcn5QUA), [meta-review](https://openreview.net/forum?id=2hOQcn5QUA&noteId=A96OeMhN3X).

Reviewers valued studying uneven acceleration but challenged narrow initial experiments, missing simple alternatives, and limited novelty in mathematical bounds. The meta-review records expanded languages, larger targets, and added comparisons, while retaining contribution concerns.

**RelaySpec inference:** a theorem that repackages a known inequality is not automatically an additional contribution. Test the plain alternative first, and connect any mechanism analysis to a useful decision or a reliably measured behavior. [Simple-baseline and theory review](https://openreview.net/forum?id=2hOQcn5QUA&noteId=v3f7vr2vUs).

## Paper structure worth learning from

These are editorial inferences, not phrases to copy or universal acceptance rules.

| Pattern in the source paper | RelaySpec adaptation |
|---|---|
| PARD introduces target dependence, its parallel-drafting solution, training cost, and practical evaluation in one connected argument. | Introduce the inherited source dependency, show the connector that replaces it, then establish cost and performance. [Full paper](https://arxiv.org/html/2504.18583v4) |
| DFlash separates the cost of drafting from tokens advanced, then tests model tasks, serving, context length, and component choices. | Explain how a cheaper round can outweigh a shorter accepted prefix. Use measured round breakdowns and a dedicated serving section. [Full paper](https://arxiv.org/html/2602.06036v2) |
| EAGLE-3 motivates architecture through a specific scaling failure and tests its changes. | Start with the measured source-trunk cost; test interface normalization and information choices under controlled budgets. [Published paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html) |
| OmniDraft connects incompatible vocabularies and adaptation to an explicit deployment setting. | Specify who already owns a drafter and needs to serve a new target; distinguish a reused drafter from an unchanged connector. [Paper version examined](https://arxiv.org/html/2507.02659v1) |

## Additional recent work affecting novelty and baselines

- [DFlash, ICML 2026 camera-ready record](https://arxiv.org/abs/2602.06036): the underlying parallel drafter, not evidence that RelaySpec preserves its outputs. Its paper was examined; a public review discussion was not verified in this study.
- [Steering Pretrained Drafters, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40255): target-informed steering is close prior art. Include its frozen-drafter variant in the taxonomy and a faithful compatible comparison; changing a method's architecture creates a RelaySpec ablation, not an official baseline reproduction.
- [SpecRoll, August 2026 preprint](https://arxiv.org/abs/2608.04962): fast state corrections plus slower updates directly overlap the proposed future online direction. This is not an accepted ICLR 2026 paper.
- [Functional Alignment Can Mislead, ICML 2025](https://proceedings.mlr.press/v267/smith25a.html): successful connections between models do not by themselves establish shared internal meaning. Use proposal behavior and held-out interventions to support RelaySpec's mechanism.
- The earlier [submission review](../../reports/ICLR_SUBMISSION_REVIEW_2026-09-05.md#5-related-work-map-and-what-it-changes) also maps PARD-2, EDA, VSD, DraftOPD, OnlineSPEC, and SPEED-Bench. These are secondary priorities for this execution window, with publication status recorded there.

## Consequence for the research program

The strongest direction is **measured low-cost retargeting of an existing frozen drafter**, with correct target verification and an explicit cost–performance tradeoff. The essential new comparison is against other inexpensive adaptation choices. One credible second-family result and one practical engine test are more valuable than many nearly identical rows. Added architecture is conditional on controlled evidence that it improves this tradeoff. The execution and framing plans translate these conclusions into concrete work:

- [Execution plan](../plans/2026-09-05-relayspec-evidence-execution-plan.md)
- [Paper framing plan](../plans/2026-09-05-relayspec-paper-framing-plan.md)
