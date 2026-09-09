# DGX Spark: essential native decoding replication

Target host: `sarcs@10.1.73.110` (`edgexpert-bc70`), NVIDIA GB10, compute capability 12.1. SSH is connected; model transfer and isolated setup are in progress. The supplied Tailscale address is unreachable from this workstation, so deployment uses the working LAN address. No credentials are stored in this bundle.

Four fixed methods: original DFlash, selected joint compact linear drafter, compact + DDTree47, and full drafter + DDTree63. This checks whether the L40S conclusions survive deployment on Spark. No training, new selection or hyperparameter sweep is planned.

`smoke.json`: four requests, one per workload, cap 128, one timing pass, plus AR outputs. `main.json`: 32 requests, eight per workload, cap 2048, two balanced timing repeats, plus one AR quality generation per request. The main manifests use the first eight requests per workload from the completed L40S confirmation. They are previously evaluated requests, not a new untouched holdout. All methods run sequentially on one visible GPU. Timing includes prefill and decoding; loading, warmup, checks, telemetry setup and output serialization are excluded. Telemetry runs continuously during measurements and its overhead is shared across arms.

## Before launch

1. Run `preflight.sh` remotely and inspect GPU activity, available UMA memory, disk, driver, CUDA, architecture and existing environments. Do not terminate other users' jobs or modify system drivers/clocks/power limits.
2. Use a compatible GPU PyTorch environment already present, or an isolated supported Spark environment after inspecting the host. Pin Transformers 5.3.0. Record Torch/CUDA differences from L40S; cross-host ratios are not pure hardware comparisons if software differs. Verify an actual BF16 CUDA matrix operation before loading models.
3. Cache the exact target and drafter revisions from `run_lane.py`, check out the exact DFlash source commit there, and copy `checkpoints/selected-29162-lane0.pt` with its frozen SHA256. Reuse existing caches when available. Check storage before transferring weights.
4. Copy the native source and Spark subdirectory into a new immutable remote directory. The runner checks all original frozen root Python hashes, official model.py hash, checkpoint hash and manifest hash.

From the native source root, with DFLASH_SOURCE and HF_HUB_CACHE set:

```bash
CUDA_VISIBLE_DEVICES=0 python hardware/spark/benchmark.py --config hardware/spark/smoke.json --checkpoint checkpoints/selected-29162-lane0.pt --output hardware/spark/runs/smoke
CUDA_VISIBLE_DEVICES=0 python hardware/spark/benchmark.py --config hardware/spark/main.json --checkpoint checkpoints/selected-29162-lane0.pt --output hardware/spark/runs/main
python hardware/spark/analyze.py hardware/spark/runs/main
```

Only launch main after the smoke's `complete.json` passes and the real runtime establishes an acceptable time estimate. Run it as a bounded, logged background process with a recorded PID; timeout and progress thresholds depend on observed Spark speed. The runner never overwrites an output directory. Runtime failures preserve partial data and do not trigger silent retries.

## Measurements

Per generation: TPS, tokens, elapsed time, repeat identity, exact native/AR agreement, acceptance lengths, truncation, PyTorch peak allocated/reserved memory and incremental allocated memory. Raw completions permit the existing isolated local math/code scoring after copying results back.

At one-second intervals: GPU and memory utilization, device-reported power/limit, temperature, SM/memory clocks, pstate, exposed GPU memory counters, system memory/swap, process RSS/high-water mark, CPU counters and load. Unsupported readings remain N/A. Integrated device-reported energy is approximate and is not wall-socket energy. Occupancy, achieved DRAM bandwidth and per-kernel timings require a separate profiler; this bundle does not claim to measure them from utilization counters.

Spark uses unified memory. NVIDIA documents that nvidia-smi may not expose GPU memory usage: https://docs.nvidia.com/dgx/dgx-spark/known-issues.html. System available memory and Torch allocations are therefore reported separately. All models remain resident for paired timing, so measured total allocator peaks are not isolated deployment footprints. No artificial memory-saving claim should be based on those peaks.

Relevant setup guidance: https://build.nvidia.com/spark/pytorch-fine-tune/run-two-sparks and https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html. The host runs driver 580.159.03. A private Transformers 5.3.0 environment uses the existing PyTorch 2.13.0+cu130 packages read-only; BF16 CUDA execution passes. This differs from the L40S runtime and must be disclosed.

## Separate hardware profiling

Nsight Systems 2025.3.2 and Nsight Compute 2025.3.1 have both captured an actual BF16 kernel on this GB10. GPU performance counters are accessible without changing permissions. CPU perf-event sampling is restricted; CUDA/NVTX tracing does not require changing that policy.

`--profile` enables a separate warmed diagnostic pass. NVTX identifies each method, target prefill, target verification, and drafter calls. Annotation wrappers must reproduce uninstrumented tokens and acceptance lengths exactly. `profile.json` uses one request at a 256-token cap; `counters.json` uses a 64-token cap. Neither produces benchmark TPS. The four-arm throughput comparison remains unprofiled.

Use the installed 2025.3.2 Nsight Systems binary, not the older `nsys` on PATH:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop -o timeline python hardware/spark/benchmark.py --config hardware/spark/profile.json --profile --checkpoint "$CHECKPOINT" --output profile-run
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum --format csv timeline.nsys-rep
```

Then inspect the timeline to select expensive kernels. Nsight Compute should profile a bounded number of kernels under `target_decode/` or `draft/` ranges, with `--profile-from-start off --nvtx --kernel-name-base demangled --set basic --section MemoryWorkloadAnalysis --launch-count 12`. Preserve the range and kernel filters in the report. Compare original DFlash against full DDTree63 first. Save `.nsys-rep`, `.ncu-rep`, CSV summaries, commands, tool versions, and raw telemetry. Report occupancy and memory metrics only for the sampled kernels; do not extrapolate their counter values to the complete workload. Nsight counter replay changes execution time and must never be treated as speedup evidence.

CUDA API durations, kernel durations, and projected NVTX spans can overlap. They must not be added to claim a total wall-time breakdown. Derive GPU-active union time from the actual trace intervals if reporting idle fractions. Metrics that the hardware does not expose remain unavailable.

The deployed bounded worker is `pipeline.py`, with status in `/home/sarcs/native-spark-20260909/evaluation-01/status.json`. It waits for the existing transfers, verifies every target/draft/checkpoint weight SHA256, and runs smoke → separate timeline → repeated main evaluation → four bounded counter passes (original/full-tree × drafting/verification). Every stage has its own log, command record, and timeout; GPU overlap or a correctness failure stops the worker. `collect.py` monitors completion and copies the result directory back to `runs/gb10-20260909/results`. No training jobs are launched by this workflow.

`preflight/` contains synthetic tool-validation results only. Its matrix-vector measurements must not be presented as inference results. The GB10 test exposed occupancy, SM, and cache counters but not DRAM bytes/second; unsupported metrics will remain null.
