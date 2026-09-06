| Role | Seed | Records | Train error | Validation error | Retention (95% request CI) |
|---|---:|---:|---:|---:|---:|
| native | 1729 | 16 | 0.0009 | 0.2332 | 66.7% [62.4,70.9] |
| native | 1729 | 128 | 0.0248 | 0.0428 | 97.2% [94.1,101.5] |
| retargeted | 1729 | 16 | 0.0017 | 0.6823 | 46.3% [43.0,49.9] |
| retargeted | 1729 | 128 | 0.1297 | 0.3165 | 87.7% [84.4,90.0] |
| native | 1730 | 16 | 0.0009 | 0.2332 | 67.0% [62.6,71.1] |
| native | 1730 | 128 | 0.0257 | 0.0433 | 97.9% [95.2,100.7] |
| retargeted | 1730 | 16 | 0.0017 | 0.6823 | 47.1% [43.7,51.0] |
| retargeted | 1730 | 128 | 0.1312 | 0.3159 | 88.9% [85.8,91.2] |

Eight shared exposed questions. Intervals resample requests within each seed; seeds do not add independent questions. Original role-specific teacher, output width and normalization are retained. References are the original512-record two-layer maps. Feature errors are not comparable across roles.
