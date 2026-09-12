128 requests per target, one active request per GPU, four paired32-question shards, one timing pass and existing fitted checkpoints. TPS is pooled tokens/summed request seconds, not sum of four GPU throughputs.

Accepted proposals exclude verifier bonus tokens. Engine scheduled-to-first-token and first-to-last-token intervals are reported only where recorded; these are not isolated GPU-kernel timings.

| Target group | Method | TPS | Mean request s | / AR | / Native | Accepted proposals/block | Exact AR |
|---|---|---:|---:|---:|---:|---:|---:|
| q8 | ar | 24.39 | 39.320 | 1.000 | 0.144 | - | 128/128 |
| q8 | native | 169.08 | 5.673 | 6.932 | 1.000 | 6.383 | 128/128 |
| q8 | five_maps-auf | 170.79 | 5.616 | 7.002 | 1.010 | 6.100 | 128/128 |
| q8 | five_maps-ce | 167.88 | 5.713 | 6.882 | 0.993 | 5.976 | 128/128 |
| q8 | dense_fusion-auf | 179.57 | 5.341 | 7.362 | 1.062 | 6.471 | 128/128 |
| q8 | dense_fusion-ce | 176.07 | 5.447 | 7.218 | 1.041 | 6.324 | 128/128 |
| q8 | original | 172.88 | 5.548 | 7.087 | 1.022 | 6.192 | 128/128 |
| cross | ar | 24.54 | 32.872 | 1.000 | 0.205 | - | 128/128 |
| cross | native | 119.45 | 6.752 | 4.868 | 1.000 | 4.029 | 128/128 |
| cross | five_maps-auf | 143.56 | 5.618 | 5.851 | 1.202 | 5.075 | 128/128 |
| cross | five_maps-ce | 136.55 | 5.907 | 5.565 | 1.143 | 4.794 | 128/128 |
| cross | dense_fusion-auf | 123.42 | 6.535 | 5.030 | 1.033 | 4.243 | 128/128 |
| cross | dense_fusion-ce | 117.32 | 6.875 | 4.781 | 0.982 | 3.985 | 128/128 |
| cross | original | 107.67 | 7.491 | 4.388 | 0.901 | 3.579 | 128/128 |
