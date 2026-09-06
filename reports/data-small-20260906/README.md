# Small-data extension, NuminaMath

Training job28337 fits dense mappers with16/32/64/128/256 distinct training examples, seed1729,8192 updates, batch4, learning rate0.0006 and the same1024 validation records as the completed large-data campaign. Existing frozen features and the verified GPU-cache access path are reused. Four fits execute concurrently; the fifth follows on GPU0 within the same four-GPU allocation. No extra fitting replicate is introduced to fill idle GPUs.

Pilot28338 follows successful fitting. Full evaluation28339 follows the pilot, re-evaluating all11 available dataset sizes from16 to32768 together, plus AR/native-DFlash/source-reuse controls:128 questions ×14 methods =1792 measurements, with a2048-token generation cap. The historical MATH-only curve is not mixed into this Numina curve. Existing large-data and512/2048-example checkpoints are reused.

Scoring and graph collection run in the persistent user service relayspec-small-collector-28337.service, bounded to4hours. jobs.json records exact commands and paths; collector-status.json records observations. Completed full results produce results.json and distinct-record-scaling.png/pdf. Failed dependency gates stop promotion; a stopped or failed job is not a completed point.

Estimated total remaining at launch: roughly50–60minutes, mostly full-answer evaluation. This is an estimate rather than a deadline.
