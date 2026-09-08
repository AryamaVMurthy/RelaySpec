# Native method research decisions

## Scope

This campaign investigates faster native speculation with Qwen3-8B and its own released, five-layer, 1.049B-parameter DFlash drafter. No source model is being transferred. These are native algorithm/architecture probes on a pretrained native system; they are not a newly trained speculative model or evidence of a novel method yet.

Every job reserves four L40S GPUs on Turing node07, with independent single-GPU lanes and 540-second per-lane timeouts. Jobs are sequential or have Slurm dependencies, so the campaign does not use more than four GPUs concurrently. All candidates use the same target verification routine. The throughput timer includes the whole generation call, including target and draft prefill. Original parameters remain resident to permit paired comparisons; no isolated memory-saving claim is made.

## Decisions made from completed experiments

1. **Job 29135: locate dispensable native computation.** Two-request, 64-token smoke screen. Removing single draft blocks, removing MLPs, and low-rank MLP replacements all lose speed. A window larger than the short context is effectively a no-op; its apparent 0.8% gain is not promoted.
2. **Job 29137: test whether more calibration resolves the linear-MLP failure.** 64 calibration records, 48 fitting and 16 held-out records, eight requests with 512-token caps. Whole-MLP replacements retain 76.5–86.4% of native GSM8K throughput. Rank increases improve fitting error without resolving validation error or decoding speed. Draft-only windows of 128/256 reduce accepted progress; 512 is near parity. Drop the whole-MLP replacement direction under this objective.
3. **Job 29139: engineering cancellation.** Cancelled after 61 seconds because the first hybrid implementation serialized many copies of frozen weights to home. This is not a scientific result. The 30 partial checkpoints (6.49 GB) were preserved with SHA-256 hashes and moved to node07 scratch. See `run-29139-cancelled/cancelled-checkpoint-archive.json`. Later jobs save learned state on scratch, with persistent checkpoint indices.
4. **Job 29140: retain nonlinear neurons and linearly reconstruct the omitted update.** 64 calibration records. The correction helps some heavily pruned arms (e.g. all-layer 50% retention rises from 82.9% to 86.7% of native throughput) but no arm beats native. An extra correction path has a runtime cost. Plain 90% retention in the first three MLPs reaches 99.0%, with an interval spanning parity.
5. **Job 29142: share target-context K/V projections across draft layers.** Rank 256/512 is too inaccurate. Rank 2,048 reaches 100.6% on GSM8K and 99.4% on MATH, both with intervals spanning parity. Exact matrix stacking also does not resolve a gain. “Exact” means algebraically equivalent full-rank projection: BF16 GEMM reshaping still changes some draft acceptance decisions. All eight final capped sequences match the native control in every arm. This is a conditioning-compression probe, not a new latent-attention architecture.
6. **Job 29143: fold reconstruction into retained nonlinear output weights.** 128 calibration records, 96 fitting and 32 held-out records. Widths are rounded down to multiples of 256. Fitting in the retained nonlinear activation basis allows correction to be folded into the existing output matrix, so there is no extra inference projection. Some arms recover near-native throughput, but no resolved gain. All eight capped sequences match native in every arm.
7. **Job 29144: tune the basic native block length.** Test blocks 4/8/12/24 against released block 16 across GSM8K, MATH, HumanEval and first-turn MT-Bench. None clearly exceeds block 16. Many final sequences differ when the target verification shape changes; retain output arrays and do not describe these as identical-work latency or output-quality comparisons. Native duplicate controls still agree exactly. This campaign does not re-establish the previously investigated numerical cause or AR equivalence.

## The profile changes the next hypothesis

Separate CUDA-stream event diagnostics in job 29143 show roughly 82% of elapsed time in target calls and 12% in the drafter on the first GSM8K and MATH requests. These are module stream intervals (including gaps), not a kernel-only profiler or a population-wide latency decomposition. Duplicate GSM8K profiles across GPU lanes are not independent questions.

If draft time were halved while everything else stayed fixed, this observed split would permit only about a 6.6% end-to-end improvement. Removing a quarter of some MLPs saves much less. Even modest loss of accepted progress can erase that benefit. The next experiment therefore spends more cheap draft computation to reduce the number of expensive target calls.

**Job 29147: native prefix self-conditioning.** Make one normal native proposal, embed the first 1/2/4/8 guesses as anchors, then run the same native drafter once more to refine the remaining masked suffix. Preserve first-pass anchor predictions, roll back and rebuild only the draft cache, and leave target verification unchanged. No fitting is required. Count two draft forwards per target verification. All four lanes completed. This untrained refinement also fails: throughput is 75.0–84.5% of native on GSM8K, 72.8–83.6% on MATH, 73.8–83.1% on HumanEval, and 84.9–85.8% on first-turn MT-Bench. All eight capped sequences match native for every candidate. Prefix self-conditioning did not improve accepted progress enough to pay for the extra pass; on math and code, it also reduced progress. Do not pursue more untrained prefix lengths.

## Completed round and next research decision

Seven waves completed, comprising 28 successful GPU lanes and 106 candidate/workload evaluations across 32 distinct development requests. Every lane finished within five minutes; the longest was 285.1 seconds. One additional submission was cancelled for storage engineering and is archived separately. The full allocation history is in `slurm-status.txt`. No candidate has a supported native-throughput improvement in the larger eight-request screens.

The data support a bounded distinction: low-data feature transport can work even when a frozen native drafter's prediction computation is difficult to simplify after training. They do not show that a smaller native architecture trained jointly would fail. The next justified direction is native training with a compact conditioning interface and a token-prediction/distillation objective that directly preserves accepted proposals. That training has not been run in this round. Further post-hoc reconstruction sweeps are not justified by the current results. A refinement training direction must also be distinguished from xPress and other existing causal correction work.

`SOURCE_COMMIT` records the base revision before experimental edits. Each run's `executed-source-index.json` hashes the actual submission archive, and `executed-source/` contains the exact native runner, intervention implementation, launcher and wave configuration extracted from that archive. The complete archives remain on Turing; small outputs, raw token arrays, source excerpts and checkpoint hashes are collected locally. The initial reviewed paper revision remains unchanged.

## Prior-work boundaries

The central research hypothesis is not an established novelty claim. Native projection compression, structured pruning, and refinement all have prior art:

- [EDSD, ACL 2026](https://aclanthology.org/2026.acl-long.2145/): target feature selection and native draft architecture/training changes.
- [DFlare, arXiv:2606.02091v1](https://arxiv.org/html/2606.02091v1): per-layer conditioning and larger native draft capacity (preprint).
- [SDFP, arXiv:2602.05499](https://arxiv.org/abs/2602.05499): pruning to construct speculative drafters (preprint).
- [xPress, arXiv:2608.02438v1](https://arxiv.org/html/2608.02438v1): a learned low-rank causal logit refiner with parallel Jacobi iterations (preprint, August 3, 2026). This is especially relevant to the refinement direction. Our frozen two-pass prefix probe is not a reproduction of xPress and does not establish priority for refinement or linear causal correction.

## Promotion gate

Only a clear development gain warrants a larger run. Freeze any selected configuration/checkpoint, evaluate separate requests with a 2,048-token cap, include native duplicates and repeated timing, and audit exact outputs and task quality before making a paper claim. The current eight-request sweeps are adaptive development screens; bootstrap intervals omit seed and selection uncertainty. Calibration fit/validation splits are disjoint by records, but this campaign does not claim a new global training/evaluation contamination audit. The reviewed manuscript has not been changed by exploratory results.
