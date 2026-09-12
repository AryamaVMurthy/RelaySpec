# Single-request speedup audit, 2026-09-12

Status: raw timing arithmetic verified; optimized-AR baseline sensitivity remains untested. Do not present these ratios as established gains over default optimized vLLM.

Qwen3-8B, 128 identical requests/output sequences, cap 2048, four paired GPU shards, one active request/GPU:

- AR: 122768 output tokens / 5032.938367 seconds = 24.392907 TPS.
- Native DFlash: same 122768 tokens / 726.090812 seconds = 169.080779 TPS, 6.931555x AR.
- Native: 16651 draft iterations, 7.373011 output tokens/iteration; request time amortized per iteration 43.606439 ms, versus AR 40.995523 ms/output token. These are amortized wall-time quantities, not isolated kernel durations.
- AR first-to-last token intervals sum to 5021.688979 seconds, versus native 715.197021 seconds. Time-to-first-token sums are 11.133149 and 10.776966 seconds. The gap is in decoding, not model-loading time or mismatched prefill inclusion.
- AR has zero timed-request cold retries; native has four, one per shard, with cold times preserved. Same rule is applied to both.
- Raw equality and 128-request cohort checks passed for all seven Qwen arms. No summing of four GPU throughputs.

Runtime configuration and logs confirm BF16, same target, prefix caching off, async scheduling off, compilation mode NONE, FULL_DECODE_ONLY CUDA graphs captured for AR and speculative arms, FlashAttention 2, VLLM_BATCH_INVARIANT=1. Compilation being disabled does not mean CUDA graphs are disabled. Whether these choices disproportionately slow AR needs a controlled benchmark; it is not established by this audit.

Primary literature comparisons (different hardware, workload and software):

- DFlash v2 Table 1: Qwen3-8B, greedy, Transformers, MATH-500 6.08x; seven-task average 4.86x. Main experiments use H200 unless otherwise specified. https://arxiv.org/html/2602.06036v2
- DFlash v2 Table 3: Qwen3-8B SGLang, concurrency 1, MATH-500 5.1x; HumanEval 4.2x. Appendix Table 12 concerns Qwen3.5-9B, not our Qwen3-8B: vLLM concurrency 1 gives 4.0x MATH-500 and 4.6x HumanEval.
- DFlare Table 1: Qwen3-8B greedy MATH500 6.72x; six-task average 5.46x; its DFlash baseline MATH500 6.29x. https://arxiv.org/html/2606.02091
- EAGLE-3 conclusion reports maximum 6.5x; separate optimized-serving comparisons show smaller ratios. https://arxiv.org/html/2503.01840

These publications establish that 6-7x ratios occur in some settings, not that our baseline is adequately optimized. Before a headline AR-speedup claim, compare AR/native on the same small fixed prompt set and physical GPU with default optimized execution, varying batch invariance and compilation separately where supported; preserve all existing measurements and report numerical agreement separately. No such additional GPU run was launched during this read-only audit.
