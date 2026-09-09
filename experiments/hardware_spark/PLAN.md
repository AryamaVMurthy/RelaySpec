# RelaySpec on the GB10 / DGX Spark platform: focused paper evidence

Goal: validate the paper's actual inherited-drafter transfer on GB10, collect matched speed and output checks, and explain hardware cost with separate memory and GPU profiling runs.

The earlier native-only queue is superseded after its successful smoke test. Its compact-drafter and DDTree variants are a separate study and must not be labeled RelaySpec evidence.

## Fixed experiment

- Frozen primary paper checkpoint, SHA256 `4af2e7026119207108798bdde038cfbab2f7f37ecd21fc80537a16b293eaf49d`. No fitting, model selection, or parameter tuning.
- Qwen3-4B DFlash inherited by Qwen3-8B; block 16, BF16/SDPA, greedy, non-thinking, batch one. Source trunk uses the original 34-layer configuration. Exact pinned revisions and implementation hashes are in `protocol.json`.
- Four methods: target AR, released native target DFlash, optimized source-model reuse, RelaySpec. Target/source/drafters co-resident during timing; this is not an isolated deployment memory measurement.
- 32 predeclared requests, eight each from MATH, GSM8K, HumanEval, and MT-Bench's first turn. Requests have been evaluated previously; this is a hardware replication, not a fresh generalization holdout. Math/GSM use the original paper's boxed-answer prompt suffix.
- Maximum 2,048 output tokens, two rotated timing repeats. Report actual tokens, stopping/capping, wall-clock TPS, paired request-bootstrap intervals, acceptance/progress, repeat identity, exact AR/native agreement and independent answer/code scores.

## Separate diagnostics

- Smoke: four requests, cap128, four controls; short repeated-output gates before measurements.
- Memory: fresh source-only and relay-only deployment processes on the same four requests, cap2048. Retain inherited embedding/head while removing source transformer layers. Report PyTorch peak allocated/reserved memory separately from system UMA and process RSS; all-arm timing memory cannot establish deployment savings.
- Nsight Systems: warmed four-method pass on one fixed MATH request, cap256; CUDA/NVTX labels for target, drafting, source reconstruction, and mapper. Instrumentation must preserve tokens and acceptance lengths.
- Nsight Compute: bounded projection-kernel sampling in source reconstruction, relay mapping, and target verification. Preserve filters and raw reports. Hardware counters describe those sampled kernels, not the full workload.
- One-second GPU power, utilization, clocks, temperature, available memory counters, process RSS, and system UMA. Final sample brackets last request for approximate device-energy integration. Unsupported counters remain unavailable; device power is not wall-socket power.

## Deliverables and completion gates

1. Successful real-GPU smoke and full four-arm evaluation with exact repeat checks.
2. Audited throughput/energy/quality results with raw rows, source/checkpoint/runtime provenance, and paired confidence intervals.
3. Isolated source/relay memory measurements and completed CUDA/kernel profiles.
4. Compact paper table and figure (speed/energy plus mechanism/memory), a cautious hardware-results paragraph, and a built/inspected updated paper PDF in the scaling paper worktree.
5. Compare against matched controls on GB10. Prior L40S observations are contextual unless prompt IDs, checkpoint, runtime, and timing instrumentation are aligned; the current Spark Torch2.13/CUDA13 runtime differs from L40S, so cross-host differences are not pure hardware effects.

Remote root: `/home/sarcs/relayspec-spark-20260909`. Native proof-of-profiler reports remain separately under `/home/sarcs/native-spark-20260909`.

Hardware identity is MSI EdgeXpert MS-C931 with NVIDIA GB10, not an NVIDIA-branded DGX Spark unit. The official MSI specification identifies this as a DGX Spark-platform system with 128GB unified memory: https://www.msi.com/Landing/EdgeXpert-MS-C931 (checked 2026-09-09). The source-only memory run retains just the 34 executed source blocks; its outputs are checked against the co-resident run.
