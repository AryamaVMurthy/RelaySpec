# RelaySpec GB10 hardware replication

This is the frozen Qwen3-4B DFlash to Qwen3-8B RelaySpec experiment. It is separate from the native-interface/DDTree experiment under `experiments/native_joint`.

The supplied machine identifies itself as MSI EdgeXpert MS-C931 with NVIDIA GB10. The study uses 32 fixed requests, four methods, two timing passes, and a 2,048-token output cap. `protocol.json` pins the source files, model revisions, mapper and manifest. `PLAN.md` defines the measurement scope.

## Execution and evidence

`pipeline.py` executes timing, isolated source/relay memory measurements, a CUDA/NVTX timeline, and bounded Nsight Compute kernel sampling sequentially. The script records exact commands and refuses to overlap another GPU compute process. Successful GPU stages are necessary but do not alone mean paper integration is complete.

`run.py` saves the executed wrapper, frozen protocol, model/runtime provenance, request records, generated tokens, acceptance lengths and one-second device/system samples. The main timing wrapper was snapshotted before later isolated-memory and annotation changes. Its SHA-256 matches the recorded main-run provenance. Frozen decoder files under `source/relayspec` were not modified.

Run analyses from the repository root after collecting the complete remote directory:

```bash
.venv/bin/python experiments/hardware_spark/analyze.py RUN/main
.venv/bin/python experiments/hardware_spark/score.py RUN/main
.venv/bin/python experiments/hardware_spark/analyze.py RUN/memory-source
.venv/bin/python experiments/hardware_spark/analyze.py RUN/memory-relay
.venv/bin/python experiments/hardware_spark/build_paper.py --run RUN --output OUTPUT
```

`RUN` must also contain `hardware-identity.json`, profiler CSV exports and their completion records. `build_paper.py` refuses incomplete timing/repeat gates or mismatched scored outputs. Its registry fingerprints the inputs and generated assets. Tests in `test_stats.py` check known paired ratios and energy integration, reject missing observations and repeat disagreements, and preserve unavailable metrics.

The local scorer uses `experiments/native_joint/analysis/quality.py`, the existing Qwen-Math grader under `vendor/qwen-math/evaluation`, and Bubblewrap. Private grading dependencies are antlr4-python3-runtime 4.11.1, word2number 1.1, sympy 1.12 and mpmath 1.3.0 in `/tmp/native-quality-deps`. Generated HumanEval code is run without host-home or network access and with resource limits. The reported code scores use base HumanEval tests. MT-Bench first-turn responses have output-agreement checks but no judge score. Optional `--first-pass` scores all completed first-pass outputs while timing repeats continue; the paper builder still requires the final completion gate and matching scoring-input hash.

`diagnose.py` is a separate, bounded follow-up on one measured native/AR discrepancy. Run it only after the pipeline finishes. FP32 output-head reprojection uses the captured BF16 hidden states, and must not be described as full-FP32 inference.

## Interpretation

- TPS uses synchronized generation wall time, including prefill, with no profiler in the timing pass. Model loading, warmup and text serialization are outside the timed interval.
- Bootstrap intervals resample paired requests within workload. There are eight requests per workload, previously evaluated in the paper; this is a hardware replication, not a fresh holdout.
- Timing keeps all arms resident. Deployment-memory claims use separate fresh processes and the same four requests. The source control retains only its 34 executed transformer blocks.
- Sampled device power supports approximate joules per token, not wall-socket energy. Unsupported global GPU memory and DRAM bandwidth counters remain unavailable.
- Nsight Systems GPU-projected ranges include GPU spans associated with each stage. They are not a union of kernel-busy intervals or an unprofiled wall-time decomposition. Nsight Compute replay describes at most eight selected projection kernels per source/relay path.
- Prior L40S measurements in `evidence/l40s-context.json` share prompts and mapper but differ in runtime and timing instrumentation. Cross-host differences cannot be attributed solely to hardware.
