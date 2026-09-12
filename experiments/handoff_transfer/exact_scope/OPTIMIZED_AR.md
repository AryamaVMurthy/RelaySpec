# Optimized AR reference, September 12

User requested AR with peak-performance settings after the 14-cell single-request comparison completed. This is a new reference measurement, not a replacement of the preserved raw results.

Array 32533 (four concurrent one-GPU tasks on node07) evaluates Qwen3-8B and Llama3.1-8B-Instruct. Each task reads its physical GPU UUID and selects the same 32-question shard previously measured on that card for each target. Overall: two AR model configurations, 256 requests, single pass, one active request per GPU, cap 2048, natural EOS, greedy BF16. No training, changed precision, or additional model checkpoint.

Runtime: pinned vLLM 0.28, optimization_level=3, asynchronous scheduling enabled, batch invariance disabled, no manual compilation override. Installed vLLM identifies O3 as currently equivalent to O2; this enables Inductor compilation and full/piecewise CUDA graphs. Model context 5120, max_num_seqs 8 but one synchronous generate request at a time, max_num_batched_tokens 8192, prefix caching disabled, GPU memory utilization 0.8. Kernel and attention choices use vLLM platform defaults. This uses the highest offered optimization level, not an exhaustive proof of best possible hardware tuning.

The evaluator reuses the exact original prompt IDs, warmup requests, timing boundaries, cap/EOS settings, and raw token recording. Setup, compilation, and warmup are excluded. Source snapshots, actual resolved engine configuration, GPU logs, output IDs, and cross-runtime exact agreement are retained. O3 can change floating-point arithmetic and hence greedy outputs; disagreements are recorded, not silently rejected or described as algorithmic lossiness.

CPU job 32537 runs after successful completion, validates raw cohorts/GPU pairing/runtime contracts, and reports each old method's TPS divided by optimized AR TPS, plus the separate total-request-time ratio and exact-output agreement. When outputs differ these quantities must not be called identical-output latency speedups. The older speculative runs remain under their original invariant, compilation-disabled runtime; the report explicitly labels this asymmetry.

Raw results: scratch/handoff-transfer-20260911/optimized-ar128-20260912. Final report/archive: control/outputs/optimized-ar128-20260912. Local results backup refreshes throughout the campaign and excludes compiler caches.

Validation: Python syntax checks, shell syntax check, and existing pooled-request metric tests passed. Initial live logs confirmed compilation (~33 seconds), full/piecewise graph capture, and one active request.

Reference: https://docs.vllm.ai/en/latest/configuration/optimization/
