# Corrected native joint-training comparisons

These checkpoint evaluations preserve the original FP32 positional-frequency buffers. Every checkpoint passed fresh-load decoding reproduction and native duplicate controls. They supersede the original after-training performance rows affected by whole-model dtype conversion. All are single-seed, eight-request, 512-token-cap development comparisons; intervals omit seed and selection uncertainty.

| Experiment | Updates | Native TPS | Student TPS | Student/native [95% paired interval] | Exact native |
|---|---:|---:|---:|---:|---:|
| joint_two_taps_lr2e-06 | 512 | 184.4 | 175.4 | 0.952 [0.910, 1.003] | 8/8 |
| joint_two_taps_lr1e-05 | 512 | 184.9 | 172.3 | 0.932 [0.889, 0.980] | 8/8 |
| joint_two_taps_lr5e-05 | 512 | 183.7 | 156.3 | 0.851 [0.807, 0.892] | 8/8 |
| joint_two_taps_lr0.0002 | 512 | 185.0 | 118.8 | 0.642 [0.615, 0.675] | 8/8 |
| interface_only | 128 | 181.2 | 173.5 | 0.957 [0.905, 1.023] | 8/8 |
| joint_two_taps | 128 | 180.9 | 171.5 | 0.948 [0.902, 1.005] | 8/8 |
| joint_shallow | 128 | 184.8 | 87.3 | 0.472 [0.441, 0.509] | 8/8 |
| native_finetune | 128 | 182.3 | 184.7 | 1.013 [0.978, 1.058] | 8/8 |

No final superiority/inferiority decision is made from this development subset. Larger data, teacher/objective variations, architecture controls, training seeds and independent confirmation remain in the active goal.
