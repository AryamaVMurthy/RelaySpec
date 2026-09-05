# EAGLE-3 relay architecture selection

This is a development-only selection on the same 32 fixed MATH-500 prompts.
The target, source proposer, optimizer, fit budget, loss, hardware, decoding
cap, and parameter count are held fixed.  The only intervention is whether the
target feature vector is RMS-normalized before the single linear relay.

## Architectural reason

The pinned DFlash proposer consumes an output-normalized conditioning tensor.
Its relay may therefore normalize its input and output without discarding a
scale needed by the frozen interface.  The pinned EAGLE-3 proposer instead
consumes the raw output of its linear multi-depth projection.  If `x = r u`, a
normalized-input predictor observes only `u`; its irreducible squared error is
at least `E[Var(c | u)]` whenever the required context `c` varies with `r`.
The raw-input affine map `R x` retains this amplitude while using exactly the
same number of parameters and one matrix multiplication.

## Matched evidence

| EAGLE relay | 128-step relative loss | 32-prompt speed vs source reuse | Acceptance retention | Exact source/relay outputs |
|---|---:|---:|---:|---:|
| normalized input | 0.413 | 1.035x [1.003, 1.068] | 69.0% | 32/32 |
| scale preserving | 0.266 | **1.237x [1.213, 1.263]** | **82.6%** | 32/32 |

The scale-preserving design wins both the mechanistic proxy and the registered
end-to-end selection metric.  The Amdahl model, computed from independently
timed components, predicts 1.238x for the observed acceptance; the measured
result is 1.237x.  It is therefore frozen for EAGLE-3 at all target scales.
DFlash retains its normalized relay because that matches DFlash's consumed
interface.  This is one interface-respecting rule, not two separately tuned
architectures.

Immutable inputs:

- `normalized-linear-analysis.json`
- `reports/eagle3-8b-scale-validation/analysis.json`
- the four per-rank JSONL files beside each analysis
- pilot jobs 25549 and 25555 under this directory
