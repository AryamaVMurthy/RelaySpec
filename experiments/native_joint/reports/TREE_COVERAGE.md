# Joint fitting for tree candidate coverage

Each final checkpoint is tested with DDTree47 and compared against the untouched released drafter with the same tree, as well as original single-path DFlash. Tests are eight development requests, cap512, one timing per stage. These are adaptive single-seed fits, not confirmation. The top-five hinge is a local coverage surrogate; actual accepted progress and TPS decide promotion. The full-model KL and coverage arms are matched; compact arms additionally warm-start earlier fits.

| Fit | Original TPS | Full DDTree TPS | Student TPS | / Original | / Full DDTree | Before / Full DDTree | Final validation KL | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tree_kl_control | 121.0 | 152.6 | 152.5 | 1.261 | 0.999 | 1.000 | 1.664 | 343.1 |
| tree_coverage_full | 119.4 | 151.2 | 148.4 | 1.243 | 0.982 | 0.994 | 1.661 | 350.8 |
| tree_coverage_two_tap | 121.1 | 152.8 | 144.3 | 1.192 | 0.945 | 0.942 | 1.751 | 344.8 |
| tree_coverage_four_layer | 118.6 | 153.9 | 122.1 | 1.030 | 0.793 | 0.811 | 2.158 | 319.4 |
