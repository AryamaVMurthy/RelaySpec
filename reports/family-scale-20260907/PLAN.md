# Expanded family-transfer speed campaign

User request: 8,192 / 16,384 distinct training examples, longer training,
block-size and other useful tuning, both Llama-to-Llama and Qwen-to-Llama.
Use four L40S GPUs on node07. No 90-minute cap.

## Data and fitting

- Pinned Numina source manifests, revision and exclusion audit in the existing
  `scaling-data/numina-v1/manifest-gate.json`. Hash-check manifests on compute.
- Nested 4,096 / 8,192 / 16,384 training records; common first 256 validation
  records from the separate 1,024-record validation manifest.
- Identical plain problem + solution text in both tokenizers, max 1,024 tokens
  per tokenizer. Sample at most 32 exact shared token-end boundaries per record.
- Frozen BF16 targets and drafter; capture raw block outputs with hooks,
  including the pre-terminal-normalization output for the final Qwen block.
- This is calibration on supplied solution text, not rollout reproduction.
  The 4k control distinguishes the new data/length/sampling recipe from the
  historical 4k one-pass mapper. Do not merge these curves with old MATH curves.
- Two objectives on identical caches: normalized dense interface relative MSE,
  and archived ZIP equal-weight per-layer + fused-context relative MSE.
- Fresh seed 42. AdamW, LR 0.001, zero decay, clip 1, 5% warmup and cosine.
  Batch 64 records (up to 2,048 positions), equal weight per record.
- Twelve complete passes; export epochs 1, 3, 6, 12. These are checkpoints of
  one schedule, not separately retuned training budgets. Validation never trains.
- Raw features remain in bounded, memory-mapped per-record files; resumable
  capture uses atomic replacement and checks record identity.

## Selection and inference

1. Small capture/fit/export/inference pilot must pass before large capture.
2. Capture in approximately 4k-record increments; reuse existing records.
3. Screen saved checkpoints with actual decoding, including old mapper controls.
   Validation loss is diagnostic, not the throughput selection metric.
4. Compare useful block sizes around the current optima (Llama 10, cross 16),
   then learning-rate follow-ups if the larger fits warrant them.
5. Confirm finalists against old mappers on the reserved 16-request manifest,
   with longer generation and matched AR. Report all attempts, not just wins.

Use FP32 target/mapper and matched FP32 greedy AR, TF32 disabled; BF16 frozen
drafter and source embedding/head. Require exact generated-token agreement for
every measured request before promotion. This is empirical finite-request
agreement, not a universal floating-point theorem. No target/verifier changes.

Historical reference on eight 1,024-token-cap requests: Llama 106.27 TPS vs
46.25 AR; cross 64.28 TPS vs 22.01 AR. New speedups must use matched requests,
caps, precision and hardware; do not compare raw TPS across different subsets.

## Launch history

- 28934: 512 train + 256 validation capture, four GPUs, complete.
- 28935: initial fitting pilot; dense input shape failure, ZIP succeeded.
- 28936: corrected four-lane fitting pilot, complete.
- 28937: four-lane exported-checkpoint inference pilot.

No improvement from the expanded campaign is claimed until decoding results
and correctness gates have completed.

- 28937 failed before inference due to a pilot manifest path typo; corrected.
- 28947: corrected inference pilot, all four lanes pass exact 2/2 AR agreement.
- Full dependent capture, fitting, screening, block tuning and final jobs are recorded in `jobs.json`.
- Block grids: Llama 6/8/10/12/16/24; cross 8/12/16/20/24/32.

## Adaptive optimization follow-up

After initial screens, choose the strongest new configuration in each family. Refit it at LR 0.0003 for 12 epochs and LR 0.001 for 24 epochs; screen checkpoints before block tuning. This selection uses development throughput only. Jobs 28953/28954; block job 28951 now depends on 28954.

## Evaluation efficiency

Job 28956 checks shared-model candidate evaluation against standalone job 28947: exact outputs and full acceptance trajectories must match. After its gate passes, candidate batches reuse frozen models and one AR reference, rotate method order, and preserve raw per-method rows. Final confirmation remains standalone per configuration, avoiding candidate-residency effects. Restore raw ZIP maps without input normalization; route cross-family variants through the existing cross-family verifier.

Unit checks: 14 mapper-campaign / research-transform tests pass.
