# Quick transfer comparison

User requested a few-sample throughput comparison. No reproduced mapper existed, so fit512 newly generated training records (first four128-example shards from28637) for3epochs with the archive objective, Xavier seed42, equal-example weighting and cosine schedule. Evaluate first8 separate evaluation prompts, output cap512. All modes use the unchanged archive benchmark; batch invariant, prefix cache disabled, warmups excluded. This is a reduced-data pilot, not the full reproduction.

28695 failed before inference due to UNIX socket path length. Work root shortened to /scratch/aryama.murthy/rs-q512.
28696 completed AR and native baselines; compiled capture failed strict sampled-versus-dense comparison (relativeMSE0.000123403 vs tolerance1e-6).
28699 reran capture using eager + batch invariant execution; all16 comparisons relativeMSE0.0. Failed features retained separately. Inspected representative prompt boundaries,69-token and4096-token responses, stop and length finishes; sampling CPU checks passed. No tolerance relaxed. Inference benchmark unchanged.
28700 captures512examples, fits3epochs, exports with numeric fold verification, then evaluates mapped on same8prompts.

Current baseline metrics: AR3427tokens/130.605466s=26.239331TPS; native3427tokens/19.521624s=175.548922TPS. Mapper pending.

## Completed comparison

```json
{
  "all_tokens_equal": true,
  "prompts": 8,
  "output_cap": 512,
  "training_examples": 512,
  "epochs": 3,
  "metrics": {
    "ar8": {
      "tokens": 3427,
      "seconds": 130.60546631598845,
      "tps": 26.239330532373543,
      "length_stops": 5
    },
    "native8": {
      "tokens": 3427,
      "seconds": 19.52162368502468,
      "tps": 175.5489223280593,
      "length_stops": 5
    },
    "mapped": {
      "tokens": 3427,
      "seconds": 21.450554127339274,
      "tps": 159.76277254451912,
      "length_stops": 5
    }
  },
  "mapped_over_ar": 6.088675637033007,
  "mapped_over_native": 0.9100754959119624
}
```

Training compute across epochs22.65seconds; pipeline capture/fit/export/eval job28700 completed. The reduced-data pilot is slower than native DFlash. No full-data improvement claim is supported.
