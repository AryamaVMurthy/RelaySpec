# Qwen3-8B draft-block ablation

Jobs 25477--25479 compare block sizes 8, 16, and 32 on the same frozen
64-prompt non-thinking MATH subset with exactly four RTX 6000 Ada GPUs. Each
job contains paired native AR, target-specific DFlash-8B, source-trunk
DFlash-4B, and RelaySpec measurements. All speculative methods reproduce the
source baseline sequence on 64/64 prompts at every block size.

| Block | Source tok/s (accepted) | Relay tok/s (accepted) | Relay/source | Target-specific DFlash tok/s |
|---:|---:|---:|---:|---:|
| 8 | 103.7 (5.47) | 164.9 (5.14) | **1.590x [1.567, 1.612]** | 175.1 |
| 16 | **137.7 (7.66)** | **210.1 (6.85)** | 1.526x [1.496, 1.556] | **229.9** |
| 32 | 88.7 (4.87) | 131.4 (4.24) | 1.480x [1.459, 1.501] | 207.7 |

Block 8 maximizes RelaySpec's *relative* improvement because it retains 94.0%
of source acceptance, but block 16 maximizes actual end-to-end throughput for
both RelaySpec and the source/target-specific baselines. Block 32 is poorly
matched to the released b16 proposer: source and relay acceptance both fall,
and its extra verification width does not compensate. The frozen primary
protocol therefore keeps block 16; no post-hoc block choice is used to inflate
the main results.

Raw outputs and complete paired summaries are under `reports/d8-block8/`,
`reports/d8-block16/`, and `reports/d8-block32/`.
