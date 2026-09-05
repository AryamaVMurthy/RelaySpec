# EAGLE-3 saved-output code quality

Official EvalPlus base and plus test cases; paired against the same run’s AR outputs.

| Target | Benchmark | Suite | AR passes | Relay passes | Delta (pp) | Paired 95% interval (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| 8B | humaneval | base | 143/164 | 141/164 | -1.22 | [-3.66, +1.22] |
| 8B | humaneval | plus | 133/164 | 133/164 | +0.00 | [-3.05, +3.05] |
| 8B | mbpp | base | 318/378 | 314/378 | -1.06 | [-2.91, +0.79] |
| 8B | mbpp | plus | 276/378 | 272/378 | -1.06 | [-2.65, +0.53] |
| 14B | humaneval | base | 143/164 | 143/164 | +0.00 | [-2.44, +2.44] |
| 14B | humaneval | plus | 134/164 | 134/164 | +0.00 | [-2.44, +2.44] |
| 14B | mbpp | base | 338/378 | 340/378 | +0.53 | [-1.32, +2.38] |
| 14B | mbpp | plus | 283/378 | 285/378 | +0.53 | [-1.06, +2.12] |

Relay and source reuse have identical per-task pass/fail outcomes in all eight cells. Keep the 8B regressions visible. These are retrospective, single-fit results; paired task intervals do not capture fitting-seed variation. A zero-width empirical bootstrap from identical outcomes does not establish population equality or noninferiority.
