# Fifteen-minute Llama correctness and speed pass

Goal: compare strongest existing mapper recipes for same-family and cross-family transfer; require exact greedy output agreement with the plain AR implementation under the same declared precision before selecting on throughput.

1. Four concurrent short lanes: old mapper and ZIP 20-epoch mapper for each transfer. FP32 throughout, TF32 disabled, eight identical MATH development requests, 512-token cap. Source trunk unloaded after retaining frozen embedding/head to fit each complete method on one L40S GPU.
2. Reject any lane with a token hash mismatch. Compare decoding and request TPS only among passing methods; report task-correctness separately from decoder equivalence.
3. Confirm the strongest passing recipe on a larger, preselected prompt set if time remains. Preserve all results, including losing and failing methods.

FP32 AR is the reference for FP32 speculation. It can differ from the earlier BF16 AR outputs; that change must be reported, not hidden. A finite prompt test does not prove universal floating-point invariance.

Initial job: 28882. Each lane timeout is bounded by the eight-minute Slurm limit. No training is required: use existing immutable checkpoints. Previous goal turn made progress by tracing actual failures to numerical execution; this pass tests the resulting precision intervention.
