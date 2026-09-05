> Historical method note. For current interface corrections and limitations, see
> [the submission review](../reports/ICLR_SUBMISSION_REVIEW_2026-09-05.md).

# RelaySpec method

## Goal

RelaySpec reuses an already-trained feature-conditioned speculative proposer
with a different, compatible target model. It removes the original source
transformer from the repeated decode loop without letting an approximation
approve output tokens.

Let `S` be the Qwen3-4B model whose hidden features conditioned a released
proposer `D`, and let `T` be Qwen3-8B or Qwen3-14B. We test both the parallel
diffusion proposer DFlash and the pinned DeepSpec autoregressive-chain
EAGLE-3 evaluator. The
matched cross-scale baseline runs `T` to verify a proposal, then runs `S` only
on the committed prefix to reproduce `D`'s conditioning interface for the next
cycle. Rejected suffixes and source layers after the final required tap are
never executed. Even this optimized baseline spends 38.8% of 4B-to-8B DFlash
runtime and 33.3% of 4B-to-8B EAGLE-3 runtime in the redundant source trunk.

## Interface translation

For either family, let `z_t^T` concatenate the exact target-verifier tap states
available for a newly committed token. The frozen proposer expects a context
`c_t^S` obtained by its released source-side fusion. RelaySpec learns one
affine map into that *consumed* interface, rather than predicting arbitrary raw
hidden states.

DFlash expects an output-normalized context,

\[
c_t^S = \operatorname{RMSNorm}
\left(W_D[h_t^{S,s_1};\ldots;h_t^{S,s_m}]\right).
\]

and therefore uses

\[
\widehat c_t=\operatorname{RMSNorm}
\left(R_\phi(\operatorname{RMSNorm}(z_t^T))\right).
\]

EAGLE-3 consumes the raw post-fusion linear state, so its matched relay is

\[
\widehat c_t=R_\phi z_t^T.
\]

This family rule follows from the frozen tensor contracts. If `z = r u`, an
input-normalized EAGLE predictor observes only `u` and has irreducible error at
least `E[Var(c | u)]` whenever the required context varies with amplitude `r`.
The raw-input map preserves `r` with exactly the same parameters and matrix
multiplication. On the registered 8B development split, this intervention
improves relative loss from 0.413 to 0.266, acceptance retention from 69.0% to
82.6%, and end-to-end speed from 1.035x to 1.237x.

The target, source model, and proposer remain frozen. After training, the
34-layer source transformer trunk is unloaded. Proposer-owned token
embeddings/heads remain only where required by the released interface. No
relay output can commit a token.

## Training

The preregistered Part-2 objective predicts the exact tensor consumed by the
frozen proposer with coefficient-free relative interface error:

\[
\mathcal L_{\mathrm{rel}}
=\frac{1}{N}\sum_t
\frac{\|\widehat c_t-c_t^S\|_2^2}
{\max(\|c_t^S\|_2^2,\epsilon_{\mathrm{dtype}})}.
\]

The denominator clamp is the smallest safe positive value for the accumulation
dtype, not a tuned quality threshold. If
`rho = ||c_hat|| / ||c||` and `theta` is their angle, each token term equals
`rho^2 + 1 - 2 rho cos(theta)`. It therefore measures magnitude and direction
without an arbitrary cosine coefficient. Raw MSE and the completed Part-1
mixed objective are retained only in a disjoint-development ablation.

Part 1 historically used
`MSE + 0.1 * (1 - cosine)` on 4,096 non-thinking math examples for 1,024
four-GPU steps. A subsequent historical refinement added `0.1 * KL`. The full
MATH-500 comparison did not resolve a benefit from that refinement, so neither
`0.1` coefficient is part of the new core method. These values continue to
describe the existing checkpoints and are not rewritten retroactively.

## Automatic provider calibration

RelaySpec is enabled per target–proposer–workload profile, outside the token
verification loop. Normalize source-reuse request latency as

\[
1=p_C+p_S+p_O,
\]

where `p_C` is proposer plus target-verification work, `p_S` is the removable
source trunk, and `p_O` is other work. Let `p_R` be relay work relative to the
same source reference and let `a_S,a_R` be committed tokens per cycle. The
predicted relay latency is

\[
\widehat L_R/L_S=(a_S/a_R)p_C+p_O+p_R.
\]

The automatic policy selects the relay only when this value is below one; the
conservative promotion check also requires the paired speed interval from the
registered calibration window to lie above one. Both boundaries are algebraic
no-slowdown conditions, not tuned quality thresholds. A profile is recalibrated
when the target, proposer, hardware, engine, or workload bucket changes. The
choice only swaps context providers and never participates in token acceptance.

## Greedy inference algorithm

1. Prefill `T` on the prompt to create the exact target KV cache. Extract the
   selected target taps and compute `\widehat c`.
2. Give `\widehat c` and the current token state to the frozen proposer. DFlash
   predicts a block of 16 draft tokens in parallel; the pinned DeepSpec EAGLE-3
   evaluator grows at most seven draft positions in an autoregressive chain.
3. Run the full target once over the proposed positions from the exact
   committed target KV cache.
4. Commit the longest prefix whose token at every position agrees with the
   target's greedy prediction, then commit the target correction token. Crop
   speculative KV entries beyond the committed prefix.
5. Extract the target taps for newly committed positions, translate them, and
   use the result to condition the next draft cycle.
6. Repeat until EOS or the fixed 2,048-token evaluation cap.

The approximation affects proposal quality and therefore speed. It is never
used as the verifier and never approves a token.

## Correctness scope

RelaySpec is approximate as an emulator of the old conditioning state but uses
canonical full-target block verification. In exact arithmetic, proposal state
can change efficiency but not the target rule. In BF16 software, however,
different accepted block segmentations can invoke different kernel shapes and
select different tokens at near-tied logits. Exact sequence agreement is
therefore reported as an implementation-reproducibility diagnostic rather than
used as the correctness proof. Every disagreement is retained, and official
task quality is measured independently for source reuse, RelaySpec, native AR,
and the native target proposer.

The implemented claim is temperature-zero greedy decoding. Distribution-exact
sampling would require the standard speculative rejection/correction rule and
is outside the measured protocol.

## Measured mechanism

The first rerun against the optimized committed-prefix source baseline uses 32
fixed MATH-500 development prompts. Source reuse and RelaySpec produce identical
sequences on 32/32 prompts. Request time falls from 161.466 to 108.621 seconds,
or 1.487x with paired request-bootstrap 95% interval [1.454, 1.520]. Source work
is 38.8%, relay work is 0.5% of baseline time, and acceptance retention is
91.5%. The same-run Amdahl equation reconstructs 1.486x from those components,
while equal acceptance would permit 1.619x. Thus acceptance recovery, not relay
arithmetic, is the remaining optimization target.

On the clean coefficient-free full MATH-500 runs, the 8B relay replaces a
38.8% source component with about 0.5% relay work. Acceptance retains 91.3% of
the source path, and end-to-end throughput increases from 148.3 to 220.0
tokens/s: 1.483x [1.475, 1.492]. At 14B, source work is 30.2%, relay work is
0.5%, acceptance retention is 87.5%, and throughput increases from 110.4 to
137.6 tokens/s: 1.246x [1.238, 1.253]. Source and relay have identical official
MATH accuracy and byte-identical sequences on 500/500 examples at both scales.

The historical conservative 8B refinement passed its smoke gate and then completed the
full MATH-500 selection run. It reaches 218.750 end-to-end tokens/s versus
218.642 for feature-only, or 1.0005x [0.9981, 1.0028] in a direct same-prompt
comparison. Both have identical outputs on 500/500 prompts, and mean
acceptance changes only from 6.992 to 7.006. Because the point gain is 0.049%
and its interval crosses one, the simpler feature-only objective remains the
primary method. The joint objective is a measured ablation, not an added
requirement.

For EAGLE-3 4B-to-8B, the optimized source path takes 305.947 seconds over 32
fixed MATH-500 development requests. The scale-preserving relay takes 247.382
seconds, or 1.237x with paired request-bootstrap interval [1.213, 1.263]. It
agrees with source reuse on 32/32 sequences. Source work is 33.3% of reference
time, relay work is 0.4%, acceptance retention is 82.6%, and the component-wise
Amdahl equation reconstructs 1.238x. This development result freezes a 1.238x
forecast; the 8B full test matrix is the confirmatory result.

With the same frozen rule at 14B, 32 source/relay development requests take
373.182 and 343.787 seconds respectively: 1.086x [1.068, 1.102], again with
32/32 exact paired sequences. The source share is 27.1%, acceptance retention
is 79.5%, and the same-run model reconstructs 1.090x. This freezes a 1.090x
forecast for the hash-frozen 468-prompt confirmation. The smaller benefit is predicted by the
source trunk becoming a smaller fraction of a larger verifier, which is the
intended scaling law rather than an unexplained model-size effect.

## Final cross-task mechanism validation

The frozen checkpoints were then evaluated without retuning on GSM8K-128,
HumanEval-164, EvalPlus MBPP-378, and 160 turns from 80 MT-Bench
conversations. Across two proposer families, two target scales, and four
workloads, the Amdahl point prediction matches the observed speed direction in
16/16 cells with 0.34% mean absolute relative error. Source-trunk share ranges
from 26.83% to 40.00%; relay share ranges from only 0.38% to 0.72%.

Raw RelaySpec has 1.1135x equal-cell geometric-mean speed. Its paired 95%
speed interval is strictly positive in 11/16 cells. Every 8B cell is positive;
the 14B code cells lose because acceptance retention falls below the derived
break-even boundary. EAGLE-14B MT-Bench has a positive 1.0221x point estimate
but an unresolved [0.9999, 1.0449] interval. The conservative provider keeps
source reuse for all five non-positive or unresolved cells and has 1.1354x
descriptive geometric-mean speed.

Official source/relay quality is identical in every scored MATH, GSM8K,
HumanEval, and MBPP pair. DFlash outputs agree byte-for-byte. EAGLE's BF16
textual differences are preserved in mismatch audits and do not change paired
official scores. These results are in `reports/final/BREADTH_MATRIX.md` and
`reports/FINAL_RESULTS.md`.
