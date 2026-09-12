These are raw five-second NVIDIA telemetry samples from completed evaluation jobs.
They include startup, warmup, compilation, measured passes and teardown. They are
whole-job telemetry, not isolated inference energy or kernel traces.

- `eval-32244.csv`: Qwen AR, original RelaySpec, ZIP and five-map CE reference job.
- `eval-32275.csv`: Qwen five-map AUF evaluation job.

Use `gpu_after.device_uuid` in each benchmark summary to identify its physical GPU.
The cards are L40S GPUs on node07; independent jobs can use different cards.
Do not divide whole-job energy by only the timed output-token count.
