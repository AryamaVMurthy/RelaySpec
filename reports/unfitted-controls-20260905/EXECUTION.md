# Zero-training controls and minimum-data search

The repository search found source reuse and trained-map variants, but no completed zero-training, source-free direct-feature control. Source reuse already tests the original drafter without a new map while retaining its source transformer.

Unmodified Qwen3-8B taps cannot enter the Qwen3-4B DFlash projection: five taps yield 20,480 channels, whereas the original projection accepts 12,800. Bypassing that projection also leaves an incompatible 20,480-wide context where the decoder expects 2,560 channels. The checked dimensions and exact meta-tensor exception are in `direct-shape-check.json`.

The new ten-minute pilot compares paired AR, native DFlash, the selected fitted relay, and two explicitly modified zero-training controls. `direct_slice` takes the first 2,560 channels of the highest selected target tap and applies no added linear layer. `frozen_fc_slice` slices each of the five target taps to 2,560 channels and uses the original frozen drafter projection. Both preserve the decoder's existing output normalization, original embedding/head, target verification and frozen drafter. These are deterministic width-compatibility choices, not literally unmodified direct reuse, nor an exhaustive search over zero-training transformations.

The pilot has 16 MATH-500 prompts, a 128-token cap, one warmup and five methods, yielding 80 measured rows. It is an initial speed/acceptance diagnostic. The short cap is unsuitable for a substantive math-quality conclusion. The allocation limit is ten minutes and generation has an eight-minute timeout. All four GPUs are used with one worker per GPU. Regions are not profiled, and the source transformer is unloaded after retaining the inherited embedding/head.

| Job | Experiment | Dependency | Expected running time |
| --- | --- | --- | --- |
| 27673 | Zero-training controls pilot | After current continuous job 27583 ends | Under ten minutes |
| 27675 | 64 distinct records, 1,024 updates | Successful pilot | About 22–30 minutes |
| 27676 | 128 distinct records, 1,024 updates | Successful pilot and preceding data cell ends | About 22–30 minutes |
| 27677 | 256 distinct records, 1,024 updates | Successful pilot and preceding data cell ends | About 22–30 minutes |

The three smaller data cells use the existing controlled fitting runner and repeat their nested subsets to reach 4,096 presentations. They use the same 128 evaluation requests, paired AR/native/source/relay methods, seed, objective and optimizer as the completed 512/1,024/2,048 cells. The ongoing continuous run supplies the 4,096-record cell. Evaluation cap remains 2,048 for these full data cells. No current benchmark was interrupted or overlapped. Maximum simultaneous allocation stays at four GPUs.

Decision criterion: report the smallest tested distinct set with at least 95% of the best measured relay throughput across the fixed-update sweep, together with acceptance, native retention, task scores and uncertainty. Do not call the threshold a global minimum. Single-seed findings require seed confirmation, and development-set selection requires a separate confirmatory evaluation before claiming broad optimality. If the smallest tested set passes, the lower boundary remains unresolved rather than automatically becoming 64 examples.

At launch, 512 is the smallest completed strong cell: 4.836x AR, 88.93% native throughput retained. The 1,024-record cell has the highest completed point estimate, 4.918x AR and 90.29% native retention. The 2,048-record cell reaches 4.883x and 89.82%. All three are within 5% of the best completed point estimate, and all score 105/128 versus AR 104/128. These observations do not identify a unique optimum.

Code checks: 208 CPU tests passed. Unit tests verify the exact selected channels, the inherited projection input, zero trainable control parameters and source-trunk unloading. The live generation pilot still must pass. Jobs run from an isolated source snapshot at commit 0ae9fa0, preserving the ongoing continuous experiment's code.

Collection: `PYTHONPATH=src:vendor/qwen-score-deps .venv/bin/python scripts/watch_controlled_jobs.py --jobs reports/unfitted-controls-20260905/jobs.json`. This collector handles only these four jobs and never submits or cancels GPU work. Results remain separate from manuscript evidence until reviewed.
