# DFlash objective selection

Decision frozen before full DFlash evaluation: use coefficient-free relative
interface MSE at both target scales.

## Controlled comparison

The two candidates used the same 4,096 fitting examples, 1,024 four-rank
updates, seed 1729, `6e-4` learning rate, zero weight decay, unit gradient clip,
192-token training cap, normalized full-rank linear architecture, target taps,
source model, target model, and proposer. The only scientific difference was:

- **relative:** `||c_hat-c||^2 / max(||c||^2, epsilon)`;
- **historical:** raw MSE plus `0.1 * (1-cos(c_hat,c))`.

All development measurements use the same 32 prompt identities. Each candidate
had to pass the executable complete-pair, exact-sequence, paired-speed-CI,
acceptance-retention, and Amdahl break-even gate before speed could select it.

| Target | Candidate | Relay/source speed | 95% CI | Exact | Acceptance retention | Gate |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | relative MSE | **1.5227x** | [1.4876, 1.5581] | 32/32 | 92.92% | pass |
| Qwen3-8B | MSE + 0.1 cosine | 1.5095x | [1.4763, 1.5445] | 32/32 | 92.23% | pass |
| Qwen3-14B | relative MSE | **1.2653x** | [1.2426, 1.2894] | 32/32 | 88.75% | pass |
| Qwen3-14B | MSE + 0.1 cosine | 1.2514x | [1.2287, 1.2761] | 32/32 | 88.06% | pass |

A direct prompt-paired comparison between the relay candidates, rather than
their separately measured source ratios, gives relative over historical speed
of 1.0085x [1.0005, 1.0174] at 8B and 1.0114x [1.0046, 1.0177] at 14B. Thus
the coefficient-free candidate is faster at both scales and its lower bounds
exceed one. The preregistered conditional `0.03`/`0.3` bracket is not triggered:
the historical `0.1` candidate lost, so no cosine coefficient enters the final
method.

The older 14B historical checkpoint reached 1.313x, but it differs in training
history and optimizer/objective settings and is retained only as historical
evidence. Its apparent gain does not survive the controlled refit and cannot be
attributed to the cosine term.

## Frozen checkpoints

- Qwen3-8B: `qwen3-8b-dflash4b-relay-relative-final/relay.pt`, SHA-256
  `4af2e7026119207108798bdde038cfbab2f7f37ecd21fc80537a16b293eaf49d`.
- Qwen3-14B: `qwen3-14b-dflash4b-relay-relative-final/relay.pt`, SHA-256
  `146ee718307e7f59d1ac1cdf05dea63a9622b8b7447f1b96e6168643be93a99c`.

The full and breadth configs already point to these immutable checkpoints.

## Evidence locations

- relative 8B: `reports/design-selection/dflash-relative-8b/`
- matched historical 8B: `reports/design-selection/dflash-historical-matched-8b/`
- relative 14B: `reports/design-selection/dflash-relative-14b/`
- matched historical 14B: `reports/design-selection/dflash-historical-matched-14b/`
- matched training telemetry: `reports/training/dflash-historical-matched-{8b,14b}/`

