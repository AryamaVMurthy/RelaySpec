# Active scope: single-request comparison only

User instruction on September12 supersedes the active batched-native and
Transformers queue. The later new-paper plan/scaling campaign remains deferred.
No training, seed sweep, block-size sweep or feature CE/KL runs are authorized
within this immediate comparison.

Targets: Qwen3-8B and Llama3.1-8B-Instruct. The mapped drafters in both settings
originate from Qwen3-4B. Each target has its own available native DFlash control.
There is no substituted native Llama3.2-3B baseline.

Seven methods per target:
1. AR.
2. Released target-native DFlash.
3. Original RelaySpec interface (unchanged original-MSE initializer).
4. Existing five dense matrices, token CE.
5. Existing five dense matrices, AUF.
6. Existing single dense fusion matrix, token CE.
7. Existing single dense fusion matrix, AUF.

Use the exact current128-request cohort per target and2048 maximum output
tokens, greedy natural-EOS decoding, vLLM0.28/BF16/batch-invariant runtime,
prefix caching off, one active request per GPU. Native Qwen block16, native
Llama block10; all transferred Qwen checkpoints block16. No block retuning.

One timing pass per existing checkpoint. Four fixed index-modulo-four shards
contain32 requests each; every shard runs all seven methods on one physical
GPU. AR is first and non-AR order rotates across shards. Array32473 contains
eight tasks (four Qwen, four Llama), with at most four active on node07. This
is14 method/target cells and1792 timed request outputs, not56 independent
benchmark configurations or additional fitting replications.

The evaluator saves complete token IDs, output lengths, synchronized engine
completion wall time, acceptance counters, cold-pass times, GPU identity and
telemetry. It also saves vLLM request timestamps where present. Report pooled
output tokens divided by summed request time, mean/median/p95 latency, ratios
to paired AR/native/original, accepted proposal tokens per block excluding
verifier bonus, exact AR-output agreement and cap hits. Engine stage times
must be labeled as scheduled-to-first and first-to-last token intervals, not
isolated GPU kernel durations. No timing is inferred by dividing batch time.

CPU job32478 independently audits all raw shards and archives results after
32473 succeeds. It verifies128 unique questions, correct shard membership,
checkpoint hashes, exact paired outputs and same-card method/AR measurement.
Existing batched results remain valid historical throughput evidence, but are
not mixed into these single-request ratios. Interrupted job outputs/logs remain
on disk; the old final audit32307 is withdrawn while this scope is active.
