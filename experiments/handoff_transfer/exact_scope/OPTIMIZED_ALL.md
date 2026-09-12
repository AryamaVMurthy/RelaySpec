> Update: jobs 32600 and 32604 were cancelled on user steering to resolve near-peak exact token agreement first. See [EXACTNESS_SEARCH.md](EXACTNESS_SEARCH.md). The following describes the prepared optimized campaign, not a completed result.

# Matched optimized evaluation: active scope

User instruction on September 12: re-evaluate every method with the same supported optimized settings as AR across four GPUs, end to end.

Seven arms on each of Qwen3-8B and Llama3.1-8B-Instruct: fresh AR; target-native DFlash; original RelaySpec; five dense matrices CE; five dense matrices AUF; single fusion matrix CE; single fusion matrix AUF. Transferred drafters originate from Qwen3-4B, including the cross-tokenizer Qwen-to-Llama bridge. All are previously trained checkpoints, with checksums checked against the earlier evaluation. No training or additional architecture/loss sweep.

Every arm uses pinned vLLM0.28, O3, Inductor compilation, full/piecewise CUDA graphs, async scheduling on, batch invariance off, V2 model runner, BF16 without quantization, TP1, prefix caching off, seed0 greedy naturalEOS. Context5120, max_num_seqs8 with exactly one active request, max_num_batched_tokens8192, memory utilization0.8. Each target uses its existing128 prompt IDs and output cap2048. Native Llama's checkpoint uses block10; other speculative arms use block16. This block-size difference is retained and disclosed rather than treated as a matched-block-budget ablation.

Four one-GPU tasks on node07, maximum four L40S GPUs. Each finds its physical UUID and reuses that card's prior modulo4 question subset (32 requests per target). Each task runs AR then all six methods for Qwen, then the same for Llama. Non-AR order rotates across workers. Four array tasks, 14 model/method cells, 1792 timed outputs, one timing pass. GPU telemetry sampled every5 seconds. Model loading, compilation and warmup are excluded; cold retries and output token IDs are retained.

Pilot array32593 completed successfully on all four GPUs. Each pilot ran AR and one representative speculative path on two requests cap128: Qwen native, Qwen original normalized RelaySpec, cross-family five-map AUF, cross-family single-map CE. These are compatibility checks, not paper benchmarks. Actual runtime settings matched in every pilot. Exact pilot output agreement varied, including native disagreements, so bitwise equivalence is measured rather than assumed.

Full array32600 launched after the pilots. CPU collector/archive32604 runs afterok32600. The runner and independent collector both reject mismatched actual compilation/scheduling/precision/runner settings, checkpoint changes and incorrect GPU/cohort pairing. Raw token validation and exact-output comparison are required; differing optimized-kernel outputs are recorded and not described as identical-output speedups. Report TPS ratios, separate total-request-time ratios, accepted proposals/block, mean/median/p95 latency, TTFT/first-to-last token intervals, output lengths/cap hits and exact agreement.

Raw: scratch/handoff-transfer-20260911/optimized-all128-20260912. Final report/archive: control/outputs/optimized-all128-20260912. Local backup automatically refreshes the campaign while Slurm is active and takes a final copy; compiler caches excluded. Four runtime/metric unit tests and syntax checks passed.
