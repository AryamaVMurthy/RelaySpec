# RelaySpec adapter and adaptation cost

The relay is one bias-free linear map. DFlash brackets it with parameter-free
normalization to match its normalized interface; EAGLE-3 keeps the same map
scale-preserving to match its raw interface. For
`m=5` target taps, target width `d_T`, and proposer width `d_D`, its parameter
count is exactly

\[
N_R = m d_T d_D.
\]

| Target | Shape | Parameters | FP32 checkpoint | Fraction of target |
|---|---:|---:|---:|---:|
| Qwen3-8B | `(5 x 4096) -> 2560` | 52,428,800 | 200.0 MiB | about 0.66% |
| Qwen3-14B | `(5 x 5120) -> 2560` | 65,536,000 | 250.0 MiB | about 0.47% |

The selected coefficient-free DFlash relays trained for 1,024 distributed
steps on 4,096 non-thinking math examples. Instrumented fitting took 85.6
seconds at 8B and 118.0 seconds at 14B on exactly four RTX 6000 Ada GPUs
(0.095 and 0.131 GPU-hours). The older conservative proposal-aligned
refinement took another 101.1 seconds from its feature checkpoint but did not
produce a statistically resolved full-run gain and is not part of the method.

The clean EAGLE-3 scale-preserving fits use the same examples and update
budget. Instrumented fitting takes 86.7 seconds at 8B and 118.5 seconds at 14B
on four RTX 6000 Ada GPUs: 0.096 and 0.132 GPU-hours respectively. These are
measured adaptation costs. DFlash reports roughly 800K target-generated
training examples, so the 4,096-example relay fit uses about **195x fewer**
examples. EAGLE-3 reports approximately 68K ShareGPT plus 464K UltraChat
entries with target-generated responses, about **130x** the relay count. These
are data-volume comparisons, not fabricated compute ratios. EAGLE-3's reported
16-A100/two-week training number is explicitly for a 70B head and is not
extrapolated to our 8B/14B targets.

The June 2026 HyperDFlash preprint supplies a second current scale reference:
it trains a new DeepSeek-V4-specific drafter on roughly 300K general plus 150K
task-oriented examples, with two five-epoch stages on eight H20 GPUs. Its 450K
unique-example volume is about **110x** the relay fit set. Because it reports no
training wall time and uses a different target and accelerator, only the
directly stated data and hardware budgets are compared; no compute-speed ratio
is inferred.

At deployment, unloading the no-longer-used Qwen3-4B source transformer saves
6.82 GiB of DFlash-8B peak allocated memory. Independent source-only and
relay-only EAGLE processes measure 7.80 GiB saved at 8B and 7.63 GiB at 14B,
with corresponding steady-allocation savings of 7.45 and 7.42 GiB. Thus one
target adapter is tens of times smaller on disk than the measured source-trunk
memory it replaces, before optional inference-precision serialization of the
currently FP32 adapter.

This is the relevant comparison to target-specific proposer retraining: the
existing DFlash proposer remains frozen and shared, while each new compatible
target pays only for this short adapter fit and small checkpoint.
