# ZIP objective: Llama and cross-family pilots

User-authorized experiment, launched as Turing job 28841 on four node07 GPUs.

Two fits run concurrently: released Llama-3.1-8B DFlash drafter retargeted to
Llama-3.2-3B, and released Qwen3-4B DFlash drafter retargeted to Llama-3.1-8B.
Each fit uses 512 existing calibration records, input cap 1,024 tokens,
25% of shared token-end boundaries, three epochs, 2,048-position batches,
equal-example weighting, Xavier seed 42, AdamW at 1e-3 with 5% warmup and
cosine decay, no weight decay, and gradient clipping at 1.

The archived `Context` class is imported unchanged. Only five bias-free
per-layer maps train. Source fusion and normalization are frozen. Loss is
the equal-weight sum of per-layer relative MSE and fused normalized context
relative MSE. Raw block outputs are captured by hooks, including the source
Qwen final block before terminal model normalization.

For both pairs, models read identical plain problem/solution content.
Positions match only when their token end offsets are identical; overlapping
spans are insufficient. This avoids pairing different text prefixes across
tokenizers. Training inputs are existing solutions, not ZIP-style generated
rollouts. Accordingly this is an objective/optimizer pilot, not an exact
replication of the full 16,384-record ZIP recipe.

Evaluation uses the existing Transformers RelaySpec runtime on eight MATH
development prompts, cap 512 output tokens, with ordinary AR and the old
mapper re-evaluated under the same settings. Llama also includes the full
source-reuse path. The old map was trained with a different data budget and
recipe; this comparison is not an isolated loss ablation. TPS must not be
directly compared with the previous Qwen ZIP vLLM runtime result.

Validation: local explicit objective calculation, frozen-buffer checks, and
FP32 folding equivalence passed. Each fit gates export on BF16 folding
relative MSE below 1e-3. Normalized exact prompt overlap between these 512
training records and the eight evaluation prompts is zero. Evaluation
token agreement must be reported alongside speed; successful training alone
does not establish decoding correctness or generalization.

Remote source and small results: `/home/aryama.murthy/rs-zip-family-20260907`.
Large features/checkpoints: `/scratch/aryama.murthy/zip-family-pilot-20260907`.
