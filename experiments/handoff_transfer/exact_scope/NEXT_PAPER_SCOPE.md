# User-directed next paper scope — September 12

Execution order is binding: finish the current cross-family loss comparisons and
full evaluation suite first. Then derive the exact new experiment plan from the
current paper and its reports; do not invent additional experiment directions.

The next paper replaces the current adapter method with five matrices trained
with AUF. Existing comparative results must still be reported accurately; AUF
being the user's selected method does not itself establish that it wins every
comparison.

For the paper's existing scaling, ablation, hyperparameter and other study runs,
use64 evaluation requests and at most1024 generated tokens. Use one training seed
and do not expand the existing paper's studies with unsolicited sweeps.

After configuration selection, perform final evaluations with128 requests and
2048-token caps across the original paper's three or four datasets and model
settings, including Qwen same-family, Llama same-family and Qwen-to-Llama. One
main final experiment has three seed/timing runs total, never a3-by3 expansion.
The exact allocation will be stated in the post-comparison plan.

Keep training batch8 per GPU, maximum four GPUs, and efficient batched evaluation.
Avoid repeated probes and smoke runs; necessary fixes and correctness checks must
still establish that measured outputs and comparisons are valid. No feature CE,
forward KL or reverse KL studies are reinstated. The next paper's experiments
have not been launched; their exact matrix awaits completion of the current suite.
