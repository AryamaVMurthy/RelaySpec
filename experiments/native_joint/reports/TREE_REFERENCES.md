# Direct comparison with full DDTree

All three arms run in the same GPU process, with rotated order, on the same requests. These are adaptive development tests. The adapted DDTree algorithm uses the pinned released draft and target; budgets exclude its bonus root. Candidate training is additional cost, recorded in its checkpoint provenance. Paired request intervals cluster timing repeats and exclude selection uncertainty.

| Candidate | Original DFlash TPS | Full DDTree TPS | Candidate TPS | / Original | / Full DDTree [95% interval] |
|---|---:|---:|---:|---:|---|
| compact_two_tap_5layer_ddtree47 | 121.3 | 153.8 | 144.6 | 1.192 | 0.940 [0.902, 0.981] |
| compact_four_drop3_ddtree47 | 121.7 | 154.1 | 124.4 | 1.023 | 0.808 [0.736, 0.860] |
| compact_four_drop2_ddtree47 | 121.6 | 154.2 | 123.3 | 1.014 | 0.800 [0.717, 0.861] |
| ddtree47_history_s2_len15 | 121.5 | 155.7 | 151.8 | 1.250 | 0.975 [0.957, 0.984] |
| ddtree47_history_s2_len8 | 121.5 | 155.7 | 152.1 | 1.252 | 0.977 [0.959, 0.986] |
| ddtree47_history_s3_len15 | 121.9 | 156.0 | 152.7 | 1.253 | 0.979 [0.961, 0.988] |
| ddtree47_history_s3_len8 | 121.9 | 156.1 | 152.9 | 1.255 | 0.980 [0.962, 0.989] |
| compact_two_tap_ddtree191 | 121.6 | 155.9 | 141.3 | 1.163 | 0.907 [0.867, 0.943] |
| compact_two_tap_ddtree95 | 121.5 | 156.0 | 141.8 | 1.167 | 0.909 [0.885, 0.935] |
| compact_four_layer_ddtree191 | 121.7 | 155.9 | 120.1 | 0.988 | 0.771 [0.702, 0.822] |
| compact_four_layer_ddtree95 | 121.7 | 155.9 | 122.1 | 1.004 | 0.783 [0.719, 0.839] |

The compact two-tap model beats original single-path DFlash but loses to full DDTree. This comparison does not support an added benefit from joint compression.
