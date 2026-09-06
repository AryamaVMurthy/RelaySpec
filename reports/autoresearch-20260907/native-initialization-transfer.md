# Native initialization across workloads

Throughput as a percentage of the paired full repacked native projection. All compact maps use33.55M projection parameters versus83.89M for full native.

| Workload | Cropped, no fitting |16 random|16 inherited|128 random|128 inherited|512 reference|
|---|---:|---:|---:|---:|---:|---:|
|GSM8K|95.7%|66.6%|99.8%|96.7%|100.6%|99.8%|
|MATH|90.8%|69.0%|93.5%|95.5%|96.2%|97.4%|
|Code|90.6%|52.6%|92.6%|92.6%|94.2%|98.0%|
|Dialogue|98.4%|72.7%|98.4%|96.2%|98.1%|98.8%|

Eight exposed development requests per workload, two turns per dialogue conversation. Seed1729. Full reference is repacked released native projection. Same-worker comparisons retain untrained cropping and all four initialized/random16/128-record fits. Paired request intervals cluster dialogue turns. No fresh confirmation, code execution or dialogue quality claim.

Interpretation: inherited native columns rescue the severe16-record random-initialization deficit. Much performance is inherited before fitting. Calibration helps math in these screens, while code intervals cross parity and dialogue barely changes relative to cropping. This supports an initialization-dependent calibration finding rather than a universal16-record sufficiency claim.
