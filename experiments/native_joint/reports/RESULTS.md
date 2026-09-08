# Standalone native joint-training results

**Protocol correction:** original after-training rows from jobs 29153, 29154 and 29155 used rounded nonpersistent RoPE buffers and are excluded from promotion. See `protocol-issue.json`. Corrected checkpoint evaluations are recorded separately as `recovery-result.json`.

Adaptive development screens with one training seed unless explicitly varied. Intervals are paired request bootstrap intervals and exclude training-seed and selection uncertainty. Throughput includes prefill. Output identity is against the pinned native decoder, not a separate AR guarantee. The before/after native controls are remeasured to reduce time-order bias.

| Run/lane | Experiment | Updates | Stage | Student TPS | Native TPS | Student/native [95% interval] | Exact native outputs |
|---|---|---:|---|---:|---:|---:|---:|
| run-29153/lane0 | interface_only | 4 | before | 132.9 | 127.0 | 1.047 [1.004, 1.109] | 2/2 |
| run-29153/lane0 | interface_only | 4 | after | 132.6 | 127.3 | 1.042 [0.999, 1.104] | 2/2 |
| run-29153/lane1 | joint_two_taps | 4 | before | 133.2 | 128.0 | 1.040 [1.001, 1.097] | 2/2 |
| run-29153/lane1 | joint_two_taps | 4 | after | 133.1 | 128.5 | 1.036 [0.997, 1.092] | 2/2 |
| run-29153/lane2 | joint_shallow | 4 | before | 71.6 | 125.0 | 0.573 [0.503, 0.637] | 2/2 |
| run-29153/lane2 | joint_shallow | 4 | after | 71.6 | 127.8 | 0.561 [0.504, 0.614] | 2/2 |
| run-29153/lane3 | native_finetune | 4 | before | 128.8 | 128.1 | 1.005 [1.002, 1.009] | 2/2 |
| run-29153/lane3 | native_finetune | 4 | after | 131.8 | 128.4 | 1.027 [0.993, 1.053] | 2/2 |
| run-29154/lane0 | interface_only | 128 | before | 174.3 | 184.3 | 0.946 [0.905, 1.000] | 8/8 |
| run-29154/lane0 | interface_only | 128 | after | 174.4 | 182.0 | 0.959 [0.906, 1.028] | 8/8 |
| run-29154/lane1 | joint_two_taps | 128 | before | 174.2 | 184.4 | 0.945 [0.904, 1.000] | 8/8 |
| run-29154/lane1 | joint_two_taps | 128 | after | 175.0 | 184.4 | 0.949 [0.902, 1.006] | 8/8 |
| run-29154/lane2 | joint_shallow | 128 | before | 82.6 | 183.8 | 0.450 [0.418, 0.487] | 8/8 |
| run-29154/lane2 | joint_shallow | 128 | after | 87.1 | 184.3 | 0.473 [0.442, 0.508] | 8/8 |
| run-29154/lane3 | native_finetune | 128 | before | 184.6 | 184.4 | 1.001 [1.000, 1.002] | 8/8 |
| run-29154/lane3 | native_finetune | 128 | after | 186.8 | 184.6 | 1.012 [0.982, 1.047] | 8/8 |
| run-29155/lane0 | joint_two_taps_lr2e-06 | 512 | before | 174.1 | 184.2 | 0.945 [0.905, 1.000] | 8/8 |
| run-29155/lane0 | joint_two_taps_lr2e-06 | 512 | after | 175.2 | 184.3 | 0.951 [0.916, 0.992] | 8/8 |
| run-29155/lane1 | joint_two_taps_lr1e-05 | 512 | before | 172.9 | 183.6 | 0.942 [0.902, 0.996] | 8/8 |
| run-29155/lane1 | joint_two_taps_lr1e-05 | 512 | after | 172.0 | 184.5 | 0.932 [0.892, 0.978] | 8/8 |
| run-29155/lane3 | joint_two_taps_lr0.0002 | 512 | before | 174.5 | 184.7 | 0.945 [0.904, 1.000] | 8/8 |
| run-29155/lane3 | joint_two_taps_lr0.0002 | 512 | after | 114.7 | 179.4 | 0.640 [0.615, 0.669] | 8/8 |
| run-29159/lane0 | target_data512 | 512 | before | 174.4 | 184.6 | 0.945 [0.905, 1.000] | 8/8 |
| run-29159/lane0 | target_data512 | 512 | after | 172.0 | 182.4 | 0.943 [0.890, 1.001] | 8/8 |
| run-29159/lane1 | native_teacher_data512 | 512 | before | 174.0 | 184.1 | 0.945 [0.904, 1.000] | 8/8 |
| run-29159/lane1 | native_teacher_data512 | 512 | after | 178.4 | 183.1 | 0.974 [0.931, 1.023] | 8/8 |
| run-29159/lane2 | blend_data512 | 512 | before | 174.1 | 184.3 | 0.945 [0.904, 1.000] | 8/8 |
| run-29159/lane2 | blend_data512 | 512 | after | 169.1 | 180.5 | 0.936 [0.889, 0.991] | 8/8 |
| run-29159/lane3 | blend_rate5e5_data512 | 512 | before | 166.0 | 180.0 | 0.922 [0.856, 0.993] | 8/8 |
| run-29159/lane3 | blend_rate5e5_data512 | 512 | after | 167.9 | 180.6 | 0.930 [0.869, 0.997] | 8/8 |
| run-29160/lane0 | nativeKD_5layers_lr5e5 | 512 | before | 174.5 | 184.6 | 0.945 [0.904, 1.000] | 8/8 |
| run-29160/lane0 | nativeKD_5layers_lr5e5 | 512 | after | 177.5 | 184.6 | 0.962 [0.911, 1.017] | 8/8 |
| run-29160/lane1 | nativeKD_4layers_drop3 | 512 | before | 123.1 | 184.6 | 0.667 [0.639, 0.699] | 8/8 |
| run-29160/lane1 | nativeKD_4layers_drop3 | 512 | after | 141.9 | 184.8 | 0.768 [0.716, 0.823] | 8/8 |
| run-29160/lane2 | nativeKD_4layers_drop2 | 512 | before | 131.1 | 181.5 | 0.723 [0.686, 0.760] | 8/8 |
| run-29160/lane2 | nativeKD_4layers_drop2 | 512 | after | 140.1 | 182.2 | 0.769 [0.734, 0.811] | 8/8 |
| run-29160/lane3 | nativeKD_interface_only | 512 | before | 174.5 | 184.7 | 0.945 [0.904, 1.000] | 8/8 |
| run-29160/lane3 | nativeKD_interface_only | 512 | after | 180.1 | 184.2 | 0.978 [0.943, 1.011] | 8/8 |
