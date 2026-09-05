# RelaySpec acceptance-aware Amdahl analysis

This report is regenerated from the frozen cross-task matrix. It explains the
speedup boundary using measured request components rather than FLOP estimates
or a similarity threshold.

Normalize optimized source-reuse latency to one:

\[
p_C+p_S+p_O=1,
\]

where `p_C` is proposer plus target-verification cycle work, `p_S` is the
removable source trunk, and `p_O` is other work. Let `p_R` be relay time
normalized by source-reference latency and let `a_S,a_R` be the source and
relay accepted-token yields. Then

\[
\frac{L_R}{L_S}=\frac{a_S}{a_R}p_C+p_O+p_R,
\qquad
\frac{a_R}{a_S}>\frac{p_C}{1-p_O-p_R}.
\]

The second expression is the exact measured no-slowdown boundary under the
component model. No fixed cosine, entropy, or representation-distance cutoff
enters it. At equal acceptance, the ceiling simplifies to

\[
S_{\mathrm{equal}}=\frac{1}{1-p_S+p_R}.
\]

## Complete mechanism check

| Family | Target | Task | Source share | Relay share | Retention | Break-even | Predicted | Observed | Equal-acceptance ceiling |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| EAGLE-3 | 8B | GSM8K | 33.69% | 0.38% | 83.84% | 65.88% | 1.2644x | 1.2613x | 1.4994x |
| EAGLE-3 | 8B | HumanEval | 33.46% | 0.42% | 75.36% | 66.23% | 1.1345x | 1.1304x | 1.4935x |
| EAGLE-3 | 8B | MBPP | 33.56% | 0.41% | 78.28% | 66.06% | 1.1799x | 1.1753x | 1.4960x |
| EAGLE-3 | 8B | MT-Bench | 33.20% | 0.39% | 78.37% | 66.46% | 1.1745x | 1.1698x | 1.4883x |
| EAGLE-3 | 14B | GSM8K | 27.36% | 0.41% | 78.80% | 72.31% | 1.0872x | 1.0826x | 1.3691x |
| EAGLE-3 | 14B | HumanEval | 27.23% | 0.47% | 66.62% | 72.60% | 0.9194x | 0.9132x | 1.3653x |
| EAGLE-3 | 14B | MBPP | 27.26% | 0.47% | 67.44% | 72.49% | 0.9320x | 0.9256x | 1.3659x |
| EAGLE-3 | 14B | MT-Bench | 26.83% | 0.41% | 75.08% | 72.94% | 1.0286x | 1.0221x | 1.3591x |
| DFlash | 8B | GSM8K | 40.00% | 0.56% | 85.68% | 59.60% | 1.4228x | 1.4237x | 1.6513x |
| DFlash | 8B | HumanEval | 39.46% | 0.64% | 73.96% | 60.31% | 1.2203x | 1.2201x | 1.6344x |
| DFlash | 8B | MBPP | 39.57% | 0.61% | 77.84% | 60.07% | 1.2866x | 1.2868x | 1.6384x |
| DFlash | 8B | MT-Bench | 38.52% | 0.56% | 82.73% | 61.27% | 1.3408x | 1.3412x | 1.6119x |
| DFlash | 14B | GSM8K | 30.92% | 0.59% | 76.39% | 68.85% | 1.1063x | 1.1025x | 1.4353x |
| DFlash | 14B | HumanEval | 30.46% | 0.72% | 61.88% | 69.52% | 0.8926x | 0.8886x | 1.4233x |
| DFlash | 14B | MBPP | 30.59% | 0.67% | 66.10% | 69.18% | 0.9567x | 0.9541x | 1.4271x |
| DFlash | 14B | MT-Bench | 29.80% | 0.56% | 76.69% | 70.10% | 1.0917x | 1.0877x | 1.4131x |

Results:

- predicted and observed point-estimate directions agree in 16/16 cells;
- mean absolute relative prediction error is 0.34%;
- relay execution is only 0.38--0.72% of reference time;
- source removal is 26.83--40.00% of reference time;
- every slowdown occurs when acceptance retention is below its measured
  break-even boundary.

The EAGLE-14B MT-Bench point estimate is positive, but its paired 95% interval
is [0.9999, 1.0449]. It is therefore explanatory evidence for the Amdahl model,
not a statistically resolved deployment win.

## Conservative provider rule

At workload calibration time, select RelaySpec only if:

1. the component model predicts speedup strictly above one; and
2. the paired request-bootstrap speed lower bound is strictly above one.

Otherwise retain source reuse. The two boundaries have direct interpretations:
algebraic no-slowdown and positive statistical evidence. They are not tuned
hyperparameters.

This policy selects RelaySpec for every 8B task, EAGLE-14B GSM8K, and
DFlash-14B GSM8K/MT-Bench. It retains source reuse for both families' 14B code
tasks and for inconclusive EAGLE-14B MT-Bench. Across 16 equally weighted
cells, raw RelaySpec has geometric mean 1.1135x; the conservative policy has
geometric mean 1.1354x and selects no measured slowdown.

This is a workload-level calibration result. It does not claim free per-prompt
oracle routing or a production workload mixture.

## Held-out scale forecast

Before full EAGLE evaluation, registered 32-prompt development profiles
predicted 1.238x at 8B and 1.090x at 14B. The independent 468-question MATH
complement observed 1.250x [1.244, 1.256] and 1.081x [1.076, 1.087]. The
relative forecast errors are +0.9% and -0.8%.

The scale trend follows directly: with the same removable 4B trunk and a larger
14B verifier, `p_S` falls and the acceptance retention required for break-even
rises. This is why RelaySpec's greatest gain occurs when the old source trunk
is a large share of the critical path and the translated interface preserves
acceptance.

Authoritative machine-readable values are in
`reports/final/BREADTH_MATRIX.json`.
