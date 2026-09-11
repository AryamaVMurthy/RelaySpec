# Batching numerical contract

Job31319 stopped before training because one of30 supported greedy predictions
changed between padded and individual BF16 forward passes on the fitted ZIP
interface. Relative logit MSE was1.1321e-4; FP32 relative MSE was2.3831e-12.
This is retained as a failed preflight, not discarded evidence.

The revised gate requires exact official-helper and padding-perturbation parity,
FP32 relative MSE below1e-10 AND exact FP32 supervised argmax agreement, and
BF16 relative MSE below4e-4. BF16 greedy and AUF-support differences are measured
and reported rather than asserted absent. The four-record training microbatch
is fixed across AUF and CE; support comes from that actual forward. No claim of
BF16 batch-shape-invariant gradients or targets is made. Exact decoding versus
AR remains an independent required check at128 requests and2048 output cap.
