# Exact requested comparison — supersedes the broad matrix sweep

Frozen paper targets: Qwen4→Qwen8, Llama8→Llama3, Qwen4→Llama8.
Seven unique architecture/loss cells per family:
- Original RelaySpec initialization + CE (`normal_ce/ce`).
- ZIP initialization + CE on the folded fusion matrix (`dense_fusion/ce`).
- Five dense matrices + CE and AUF (`five_maps`).
- One dense final-fusion matrix + CE and AUF (`dense_fusion`; CE overlaps ZIP CE).
- Five BA rank56 adapters + CE and AUF (`five_ba56`).

No fresh dense controls, single fusion BA, native drafter baseline, extra seed,
capacity, data-size, or regularization sweep is scheduled. No new LR sweep:
same-family LRs come from already completed independent validation; cross
uses 1e-4, explicitly untuned. Existing artifacts are retained.

4096 distinct fitting records, rollout cap4096, up to32 distinct eligible
anchors per record, 2000 updates × global batch8 =16000 presentations.
BF16, frozen drafter/target, cached features, fused AdamW, length grouping,
2 GPU DDP and gradient accumulation. Short records supply fewer anchors.

Final vLLM evaluation:128 fixed requests, cap2048, greedy, natural EOS,
batch4 throughput first and batch1 latency comparison, 3 timing repetitions.
AR and unchanged original/ZIP initializers are reference measurements.
Full token and finish equality checked against AR; no final-set selection.
Adapters folded for inference. GPU utilization/memory/power logged.

Two serial dependency lanes, each at most2 GPUs: Qwen node07, Llama node06.
Each trained cell is evaluated immediately before the next fit. Cross data
and cross fits use node07 after Qwen finishes, at most2 GPUs even if Llama
continues. This enforces max4 total without waiting for unrelated studies.

## Confirmed amendments
Use32 anchors in every token-block training run;512 artifacts stay isolated.
Feature-only original-interface objectives: forward KL(source||mapped), reverse
KL(mapped||source), and soft-target CE. Treat normalized fused feature vectors
as logits with standard softmax T=1. No extra exponential or positional weights.
All three are run on the same cache/initialization/seed/schedule. Forward KL and
CE gradients are algebraically identical; the numerical unit test confirms this.
These three fits use the existing feature sampler, not block anchors. They are
three-epoch fits at1e-3 matching the reconstruction baseline's feature protocol;
not a compute-matched comparison to2000-update token training.

Total:7 token-objective cells +3 feature-objective cells per family =30 fits.
AR, unchanged original feature-MSE and unchanged ZIP feature-loss exports are
reference evaluations. Loss variants never modify the target or vocabulary IDs.
32-anchor token fits reuse learning rates selected earlier at512 anchors;
they are not newly optimized for32 anchors. Cross rates remain untuned1e-4.
