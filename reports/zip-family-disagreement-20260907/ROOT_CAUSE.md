# Root cause of traced greedy disagreements

Confirmed on four original pilot failures: two Llama-only and two
Qwen-to-Llama transfers, using the 20-epoch pilot checkpoints. Job 28875
reproduced the original job 28857 output token hashes for both AR and
speculative decoding in all four cases. Prompts include the exact benchmark
math suffix and chat rendering. These are measurements of actual failures,
not an assumption based on a small logit gap.

## Finding

BF16 numerical differences between single-token and block target execution
alter greedy selection. Small hidden-state changes interact with coarse
BF16 output-logit rounding and deterministic argmax tie breaking. The
sequence consequently diverges even though the target verifies the draft.
The implementation does not provide bitwise-equivalent target execution
between these two computation shapes.

| Target / prompt | First differing generated token (zero-based) | AR choice | Speculative choice | AR logits (AR choice, spec choice) | Spec logits (AR choice, spec choice) |
|---|---:|---|---|---|---|
| Llama-3.2-3B / number_theory/931 | 15 | ` that` | ` based` | 23.000, 23.000 | 22.875, 23.000 |
| Llama-3.2-3B / number_theory/521 | 7 | ` x` | ` $` | 22.875, 22.750 | 22.875, 22.875 |
| Llama-3.1-8B / number_theory/521 | 46 | `.` | `,` | 32.250, 32.000 | 32.250, 32.250 |
| Llama-3.1-8B / intermediate_algebra/207 | 22 | `Since` | `Step` | 21.375, 21.250 | 21.375, 21.375 |

The comparison uses identical complete token prefixes at each first
divergence. Every first divergence lies at the first prediction of a
verification block. The emitted token is the verifier's own argmax; no
unverified draft token is needed to produce these failures.

## Controlled evidence

- Exact speculative-cache/full-block replay reproduced the original
  speculative decision in all four cases.
- Replaying the original AR cache sequentially reproduced the original AR
  decision in all four cases.
- Holding the AR cache fixed, switching from single-token to full-block
  execution flipped the decision in three cases (both Llama-only cases and
  cross-family number_theory/521).
- Holding the speculative cache fixed, changing full-block to single-token
  execution flipped the other case (cross-family intermediate_algebra/207),
  as well as Llama number_theory/931.
- Thus changing the compute shape with an identical saved cache is
  sufficient to reproduce a flip for every traced case. Wrong text prefix
  or a differently trained mapper is not required to explain these flips.
- Numerical cache-history differences also affect decisions: changing the
  cache while retaining the same sequential execution changes the choice
  for Llama number_theory/521 and cross-family number_theory/521.
- Computing the output projection in FP32 on the two *observed* final
  hidden vectors yielded the same argmax in every pair. In one case it
  agreed with the original speculative choice; in three it agreed with AR.
  This intervention changes the numerical reference and is not proof of
  preserving the old BF16 AR outputs.

For Llama number_theory/521, FP32 projection gave logits `(22.91551,
22.80224)` from the AR hidden state and `(22.92597, 22.82177)` from the
speculative hidden state. Both prefer ` x`. In BF16, the second state's
two logits round to the same value, 22.875. Argmax then selects the smaller
token ID: ` $` (400), rather than ` x` (865).

## Scope and remaining work

This establishes the cause for these four reproduced failures, not a
classification of every mismatch in the eight-prompt evaluations. The
experiments do not isolate the first numerical difference to an individual
transformer kernel, nor establish that head promotion alone fixes complete
decoding trajectories. No decoder change or corrected TPS claim is made.
The original pilot speedups remain unsuitable as lossless-performance claims.

The initial exploratory trace job 28874 omitted the math-answer suffix.
Its results are retained for audit but excluded from the claim above.
Job 28875 corrected the prompt and verified every original output hash.

Evidence: `run-28875/{llama-0,llama-1,cross-2,cross-3}/diagnosis.json`,
with small tensor artifacts saved beside each JSON. `trace_source.py`
preserves the executed diagnostic. The existing token-commitment and
vocabulary-bridge unit tests passed (25 tests); those tests alone do not
establish numerical equality of real model executions.
