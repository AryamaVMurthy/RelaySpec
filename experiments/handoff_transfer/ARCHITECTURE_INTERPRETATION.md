# What the adapter comparisons isolate

Let d_s be the source drafter width, d_t the new target width, and split frozen
fusion F=[F_1 ... F_5], with F_i in R^(d_s×d_s).

Five dense maps deploy as W=[F_1 M_1 ... F_5 M_5]. One unrestricted fusion map
learns W directly. Both train 5*d_s*d_t parameters. If every F_i is invertible,
any dense W can be represented by M_i=F_i^(-1) W_i: these have equal linear
expressive capacity. This is a conditional mathematical statement; numerical
rank/conditioning of the actual fusion blocks has not been measured here.

Even under that condition their optimization differs. A gradient update to M_i
induces an effective W_i update preconditioned by F_i F_i^T. Thus different
learning-rate sensitivity or convergence does not itself establish a capacity
advantage. The five-map factorization also preserves a layerwise initializer.

A fusion residual BA with rank r has r*(d_s+5*d_t) parameters and update rank at
most r. Five independent residuals B_i A_i have 5*r*(d_s+d_t) parameters and an
effective concatenated update rank at most 5r (also bounded by output width).
Both use r=56 in the requested experiment, but they are NOT parameter-matched.
Actual fitted ranks can be lower than these bounds.

| Pair | Dense / five dense | Fusion BA56 | Five BA56 | Near-matched fusion rank |
|---|---:|---:|---:|---:|
| Qwen4 drafter → Qwen8 | 52,428,800 | 1,290,240 | 1,863,680 | 81 |
| Llama8 drafter → Llama3 | 62,914,560 | 1,089,536 | 2,007,040 | 103 |

Rank81 Qwen and rank103 Llama are possible *additional* parameter-matched
controls, not replacements for the requested rank56 conditions and not runs
already performed. Do not claim matched-size superiority from the rank56 pair.

The original normalized RelaySpec interface differs from raw ZIP fusion in
input normalization as well as initialization. Its CE continuation is a useful
method comparison, but does not isolate initialization alone. ZIP-initialized
versus fresh raw dense fusion isolates warm-start initialization more directly.

The final paper should report trainable parameters, initializer cost, token-loss
fit cost, and deployment shape separately. Folding makes dense deployment size
identical across these raw linear variants; fewer trainable parameters does not
by itself reduce inference FLOPs. Any throughput difference at the same folded
shape should be explained through proposal quality/accepted progress or measured
runtime differences, not asserted to be a smaller deployed model.
