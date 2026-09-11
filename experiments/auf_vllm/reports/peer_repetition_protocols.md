# Training seeds and evaluation repetitions in related papers

Checked2026-09-11. This is a literature audit and proposed protocol revision, not a report of new experiments. The current campaign's3-training-seed×3-timing-repeat budget was a conservative design choice, not a requirement established by these papers.

## Sources and findings

| Paper | Version checked | Training-seed replication | Evaluation repetitions |
|---|---|---|---|
| PARD, ICLR2026 | Official18-page proceedings PDF, Section4.1, AppendixC, result tables | No independent multi-training-seed count stated in the checked paper | No explicit timing-repetition count or3×3 design found; reports TPS/speedup on benchmarks |
| EAGLE-3, NeurIPS2025 | Official20-page proceedings PDF, experiments and checklist | No training-seed replication count found | No3×3 protocol found; checklist item7 explicitly answers No to statistical-significance reporting |
| DFlash, ICML2026 | Camera-ready arXiv2602.06036v2, including evaluation and appendices; acceptance status on author arXiv page | No training-seed replication count found | No explicit timing-repetition count or3×3 protocol found |
| SD², AAAI2026 | Published decoding-parameters text and author extended arXiv2511.09844, Section4 | The reported three seeds concern generation, not three independently trained checkpoints | Three generation seeds at temperature1; mean/std reported for stochastic sampling. This does not establish three timing repetitions at greedy temperature0 |

Sources:

- [PARD official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/ea17f1a1685d65e66af19da44471b9ae-Paper-Conference.pdf).
- [EAGLE-3 official proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/c7b5a35ea98b62512a869c19ea7b03cb-Paper-Conference.pdf), checklist item7 on PDF page17.
- [DFlash camera-ready paper](https://arxiv.org/html/2602.06036v2); [author acceptance/version record](https://arxiv.org/abs/2602.06036).
- [SD² published paper](https://ojs.aaai.org/index.php/AAAI/article/download/40255/44216); [author extended paper](https://arxiv.org/html/2511.09844).

Absence of a stated count is not evidence that authors ran exactly once. Searches for seed/repeat/statistical terms were combined with inspection of experimental setup, appendices and, for EAGLE-3, the published checklist. The findings concern what these papers report; undocumented execution cannot be inferred. EAGLE-3's checklist justification is the authors' statement, not a recommendation to omit uncertainty analysis.

## Distinctions needed in our study

- Training seed: independent optimization realization, including initialization and batch/anchor order; expensive because it requires fitting another checkpoint.
- Timing repetition: benchmark the same checkpoint/prompts again to measure hardware/runtime noise.
- Generation seed: change stochastic token sampling. It is not a substitute for either of the above. Our principal temperature0 experiments do not need three sampling seeds merely because SD² used them at temperature1.
- Request bootstrap: quantifies request variation conditional on the fitted model and runtime observations. It cannot establish training-seed stability or eliminate timing noise.

## Recommended replacement for the blanket3×3 policy

1. All exploratory data/epoch/rank/MLP/block-size endpoints: one fit seed and one warmed full128-request evaluation at2048 cap. Retain all endpoints. This was already the intended policy for many sweeps; do not claim a new9× saving there.
2. Principal frozen transfer comparisons: one declared fit seed, three rotated timing measurements, with the matched AR/native/ZIP/CE/AUF controls. Training-seed robustness is assessed separately below.
3. Training-seed confirmation: predeclare Q8 and L3 at the main fixed data/configuration, add two fit seeds for ZIP/CE/AUF, and evaluate each extra seed once on a fixed128-request workload. Together with the first seed this gives three fits without timing every fit three times. Include both successful and unsuccessful confirmations, with no best-seed selection.
4. If timing noise prevents resolving the main effect, use a predeclared bounded additional measurement policy or report an unresolved difference. Do not keep rerunning until a result becomes significant. Additional timing applies to the paired controls as well.
5. Preserve128 requests,2048 output cap, exact-token checks, quality scoring, and same-runtime baselines. Reduced repetition changes the uncertainty scope, not these core evaluation requirements.

This proposal supports timing reliability broadly and training stability in representative cases. It does not claim all model/workload cells have three independent fits. Error bars must identify what was resampled.

## Consequence for the prior budget

The old principal trained-method count was4pairs×4workloads×5variants×3fitseeds×3timingrepeats=720 units of128 requests (92,160 answers).

Keeping the same five variants and four pairs for illustration, one fit seed and three timings gives240 units. Adding two extra seeds for three core losses on two predeclared pairs and one workload adds12 units, totaling252 units (32,256 answers), a65% reduction in this component. Cross-family feasibility and reducing the optional LoRA track change counts further and must be accounted for separately.

This is not a65% reduction in the entire project's elapsed time: integration, data generation, existing single-seed exploratory sweeps, and manuscript work remain. The earlier10–18day estimate covered a deliberately larger protocol, not a conference-mandated minimum. Recompute the complete launch matrix after finalizing this repetition revision and measuring AUF throughput.
