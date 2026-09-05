# Exact-output target-feature relay: four-GPU probe

Date: 2026-08-26

## Result

On four paired non-thinking Math-500 prompts with three timing repetitions,
Qwen3-14B plus the trained Qwen3-4B DFlash-b16 proposer decoded at 90.67
tokens/s with the normal 4B source trunk. Replacing that trunk with the
target-feature relay decoded at 116.06 tokens/s, a measured 1.280x end-to-end
speedup.

All 12 relayed timing outputs matched their full-DFlash token sequences exactly.
The target still performs ordinary full speculative verification, so the
relay can affect acceptance and speed but does not define emitted tokens.

| Metric | Normal DFlash | Target relay |
|---|---:|---:|
| Aggregate decode tokens/s | 90.67 | 116.06 |
| Speedup | 1.000x | 1.280x |
| Mean accepted tokens/cycle | 6.371 | 5.575 |
| Exact sequence matches | 12/12 | 12/12 |
| Source-trunk or relay time/cycle | 20.07 ms | 0.294 ms |

The relay is a single dense projection from five already-computed Qwen3-14B
hidden-state taps into the 2,560-dimensional conditioned state expected by
DFlash. It was trained for one pass over 4,096 MATH examples and one
low-learning-rate refinement pass, with both model backbones and DFlash
frozen. Each four-GPU pass took about 117--118 seconds after model loading and
peaked at 40.33 GB allocated memory per GPU.

## Scope

This is a fast mechanism probe, not a paper-scale benchmark: four prompts,
three timing repetitions, 128 generated tokens, temperature zero, Qwen3
non-thinking mode, block size 16, and one RTX 6000 Ada per prompt. The result
establishes implementation viability, exact-output behavior on the paired
sample, bottleneck removal, and a repeatable positive speed signal. Larger
held-out evaluations remain necessary for a publishable claim.

Raw summaries:

- `raw/relay-probe-clean-main.json` (fresh clean-tree reproduction: 1.241x,
  12/12 exact)
- `raw/relay-probe-128step.json`
- `raw/relay-probe-1024step.json`
- `raw/relay-training-1024step.json`
- `raw/relay-probe-refined-single.json`
- `raw/relay-probe-refined-repeat3.json`
- `raw/relay-training-refined.json`
