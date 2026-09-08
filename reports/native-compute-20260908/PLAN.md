# Native speculative decoding research

Question: can RelaySpec's finding that useful prediction survives inexpensive linear transformations be applied inside a native speculative decoder to remove expensive computation? This campaign keeps Qwen3-8B as both the native drafter's conditioning model and verifier. It does not retarget a drafter across models.

First wave has four independent GPU lanes on node07, each externally capped at 540 seconds:

1. Remove individual native DFlash blocks 1–4. Locate the depth/accepted-progress tradeoff before fitting replacements.
2. Remove individual MLP residual updates, layers 0–4, retaining every attention block. Test where nonlinear compute is dispensable.
3. Fit rank-64/256 linear replacements for MLPs in layers 3/4, using actual native draft activations from Numina training-prefix continuations. Fit and validation are split by entire calibration records. Compare end-to-end speed and progress to native and an identical native duplicate.
4. Restrict draft attention to four sink positions plus 64/128/256 recent positions. Target attention and verification remain unchanged. This first implementation retains the full stored draft cache and tests computation only, not cache-memory savings.

The first smoke test uses two exposed GSM8K development questions, a 64-token output cap and four calibration records. Only after implementation/control gates pass will the next wave use eight or more requests and larger output caps. At least two of the four lanes remain under ten minutes; the initial policy bounds all four. New candidates are compared in rotated per-request method order with one warmup per candidate. No change to the target acceptance rule is permitted. Exact output arrays and acceptance trajectories are retained. BF16 block/AR differences remain a separate numerical issue, and parity with a native baseline does not establish AR identity.

Promotion requires actual native throughput gain, not just reduced parameters, reduced feature loss or comparison against a transferred mapper. Retain all failed and negative experiments. Freeze a promising choice before separate requests, longer answers and repeated timing. Calibrating a native released drafter is an experimental starting point for a new computation design, not evidence that a new architecture has been trained from scratch.

## Prior-work boundaries

- [EDSD, ACL 2026](https://aclanthology.org/2026.acl-long.2145/) already studies entropy-driven feature-layer selection and architecture/training changes. Fewer target taps alone is not a new native method.
- [DFlare, June 2026 preprint](https://arxiv.org/html/2606.02091v1) studies per-layer target fusion and increased draft depth. Its native architecture evidence motivates examining where native draft computation is necessary, but does not establish our proposed linear replacements.
- [DFlash official project](https://z-lab.ai/projects/dflash/) supplies the frozen native baseline. The same pinned implementation and checkpoint are used throughout this wave.
- Existing RelaySpec native compression, initialization, residual and gain studies are preserved. They mostly establish retained throughput with cheaper projections and do not by themselves prove a native speed advantage.

Source identity, pinned models, environment, GPU telemetry, request manifests, calibration file hashes, fit times, train/validation losses and raw capped outputs are recorded per job. No exploratory claim is inserted into the reviewed paper before it is supported.
