# Direct comparison with full DDTree

All three arms run in the same GPU process, with rotated order, on the same requests. These are adaptive development tests. The adapted DDTree algorithm uses the pinned released draft and target; budgets exclude its bonus root. Candidate training is additional cost, recorded in its checkpoint provenance. Paired request intervals cluster timing repeats and exclude selection uncertainty.

| Candidate | Original DFlash TPS | Full DDTree TPS | Candidate TPS | / Original | / Full DDTree [95% interval] |
|---|---:|---:|---:|---:|---|
| compact_two_tap_5layer_ddtree47 | 121.3 | 153.8 | 144.6 | 1.192 | 0.940 [0.902, 0.981] |
| compact_four_drop3_ddtree47 | 121.7 | 154.1 | 124.4 | 1.023 | 0.808 [0.736, 0.860] |
| compact_four_drop2_ddtree47 | 121.6 | 154.2 | 123.3 | 1.014 | 0.800 [0.717, 0.861] |

The compact two-tap model beats original single-path DFlash but loses to full DDTree. This comparison does not support an added benefit from joint compression.
