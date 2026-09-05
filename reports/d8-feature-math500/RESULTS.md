# Qwen3-8B feature-relay MATH-500 result

Jobs `25430` (generation) and `25431` (official Qwen math scoring), run on
2026-08-27 with exactly four RTX 6000 Ada GPUs. All methods use Qwen3 in
non-thinking, deterministic mode with a 2,048-token output cap.

| Method | Accuracy | End-to-end tok/s | Mean accepted | Speedup vs native AR |
|---|---:|---:|---:|---:|
| Native AR | 74.8% | 44.5 | 1.00 | 1.00x |
| Naive 4B source reuse | 74.2% | 145.5 | 7.76 | 3.27x |
| Official target-specific DFlash-8B | 74.2% | 241.2 | 8.01 | 5.42x |
| RelaySpec feature relay | 74.2% | 218.6 | 6.99 | 4.91x |

RelaySpec is **1.503x** faster end to end than source reuse, with a paired
95% bootstrap interval of **[1.494, 1.511]**. Its accuracy delta versus
source reuse is exactly 0.0 points, and all **500/500** RelaySpec sequences
are byte-identical to source reuse. Its accuracy delta versus native AR is
-0.6 points, with paired 95% interval [-2.8, +1.4]. The difference from
native AR arises from the DFlash proposal path rather than the relay:
source reuse and official target-specific DFlash have the same 74.2% score.

RelaySpec attains 90.6% of official target-specific DFlash end-to-end
throughput while avoiding its target-specific source-trunk computation.
The relay is one bias-free projection from five 4,096-wide target taps to the
2,560-wide proposer interface: 52,428,800 trainable parameters. Feature
training consumed 107 seconds on four GPUs, or 0.119 GPU-hours, over 4,096
distributed examples. No target or proposer weights were updated.
Its measured request time is 83.95% target verification, 13.48% DFlash
proposal, 0.77% relay, 0.77% target prefill, and 1.03% other runtime. By
contrast, source reuse spends 38.20% in the redundant 4B source trunk and
51.53% in target verification. This directly explains why eliminating the
source trunk produces a large end-to-end gain even though RelaySpec accepts
slightly fewer tokens per cycle (6.99 versus 7.76).
If the measured 38.20% source fraction could be removed with no other change,
Amdahl's law gives a 1/(1-0.3820) = 1.618x ceiling. The observed 1.503x
end-to-end gain realizes 92.9% of that ideal; the residual gap is explained by
the acceptance decrease and relay overhead.

The gain is not an artifact of a single response-length regime. On paired
subsets, decode speedup over source reuse is 1.513x for outputs up to 256
tokens, 1.500x for 257--512, 1.508x for 513--1,024, and 1.495x for
1,025--2,048 tokens. By prompt length it ranges from 1.509x at most 80
input tokens to 1.481x above 180 input tokens. Thus the measured advantage is
stable across the MATH-500 length distribution, with the expected mild decline
for the longest prefills.

## Provenance

- Generation job `25430`; official Qwen-math scoring job `25431`.
- Target `Qwen/Qwen3-8B` at revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- DFlash source commit `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`.
- Immutable manifest SHA-256
  `415bee94c76b6241da65c0b4a951222d29906d30fe453ba663c47b4535a20f71`.
- Raw paired evidence: four rank files, 500 rows each, 2,000 total records,
  stored under `raw/`.
