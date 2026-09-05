# RelaySpec completion audit

Date: 2026-08-28

This audit treats completion as unproven until every requested element has an
authoritative current-state artifact.

| Requirement | Status | Authoritative evidence |
|---|---|---|
| Complete the plan before full execution | Proven | Frozen protocol and execution tasks in `docs/plans/2026-08-28-relayspec-rigorous-evidence-and-final-benchmarks.md`; amendment log in `configs/relayspec_protocol.yaml` records only corrections, not retrospective method tuning. |
| Implement the mechanism end to end | Proven | Shared proposer/context-provider implementation in `src/relayspec/`; DFlash and EAGLE integration tests; source/relay/full-target raw rows in all final directories. |
| Make it work for both DFlash and EAGLE | Proven | Four full MATH cells and 16 cross-task cells in `reports/final/BREADTH_MATRIX.json`; both families operate at Qwen3-8B and Qwen3-14B. |
| Ground every scientific constant | Proven | Evidence classes and rationales in `docs/research/relayspec-decision-register.md`; machine audit passes protocol version 1 over 27 active configs. |
| Use current primary-paper evidence | Proven | Publication status, primary URLs, checked revisions, and experimental implications in `docs/research/relayspec-source-log.md`, `reports/RELATED_PAPER_EXPERIMENT_REQUIREMENTS.md`, and `references.bib`. |
| Avoid arbitrary `0.1` similarity thresholds | Proven | Coefficient-free relative interface MSE is the core objective; matched objective selection is in `reports/design-selection/dflash/OBJECTIVE_SELECTION.md`; automatic provider boundaries are algebraic speedup one and paired-CI lower bound one. |
| Run complete main benchmarks | Proven | Four 500-prompt MATH directories under `reports/final/`; each paper summary contains 500 paired requests and identical paired official accuracy. |
| Run cross-task breadth | Proven | GSM8K-128, HumanEval-164, MBPP-378, and 160 MT-Bench turns for four family/target cells; expected counts are enforced by the matrix builder. |
| Correctly execute both MT-Bench turns | Proven | `reports/final/eagle3-{8b,14b}-mtbench-two-turn/` each contain 160 rows per paired method and 80 conversation bootstrap clusters. |
| Measure speed, acceptance, exactness, quality, and components | Proven | Full matrix Markdown/JSON records paired speed CIs, exact rate, survival-derived acceptance retention, cap/latency fields, and CUDA-region Amdahl terms; pinned official scorer outputs live beside raw rows. |
| Explain where speedup occurs | Proven | 16/16 Amdahl point-direction matches and 0.34% mean absolute relative error in `reports/amdahl-analysis.md`; source share 26.83--40.00%, relay share 0.38--0.72%. |
| Avoid deploying measured slowdowns | Proven | Conservative profile policy keeps source for both 14B code tasks and unresolved EAGLE-14B chat; descriptive matrix geometric mean is 1.1354x. |
| Measure target accuracy degradation | Proven | Identical paired official accuracy for every scored MATH, GSM8K, HumanEval, and MBPP cell; every EAGLE textual mismatch is retained in mismatch-audit artifacts. |
| Compare native AR and target-specific proposers | Proven | Deployment-frontier table in `reports/FINAL_RESULTS.md`; available RelaySpec/native-proposer ratios are 89.20--91.19%. |
| Measure memory and fitting efficiency | Proven | Separate-process EAGLE memory artifacts at 8B/14B; `reports/adapter-cost.md` records parameters, examples, fit time, and scoped data-volume comparisons. |
| Use exactly four GPUs for GPU jobs | Proven | Every launcher enforces four visible devices; final allocation JSON exposes devices `0,1,2,3`; Slurm ledger documents serialized execution and no GPU job remains queued. |
| Monitor and correct pipeline defects | Proven | `reports/EXECUTION_STATUS.md` records the two-turn chat fix, pinned-Transformers overlay fix, checkpoint-path fix, failed verification due to an unsynced required artifact, and successful replacements. Invalid runs produced no promoted evidence. |
| Verify code and configuration | Proven | Local 118/118 tests, Ruff, `git diff --check`, prompt-overlap audit, constant audit; remote job 25642 passes 118/118 tests on exactly four GPUs. |
| Deliver final report and proposal PDF | Proven | Eleven-page LaTeX PDF compiles with bibliography, has no undefined citations or overfull boxes, passes Ghostscript parsing, and was visually inspected on title, main-result, breadth, Amdahl, artifact, and bibliography pages. |
| Leave no required work pending | Proven | `reports/EXECUTION_STATUS.md` marks all planned jobs complete; cluster queue is empty for the study account; core final documents contain no TODO/TBD/in-progress marker. Production serving concurrency is explicitly outside the claimed batch-one scope, not an unfinished claimed experiment. |

## Immutable artifact digests at audit time

```text
e0c4043ddc792da6cb8b8c96f20153f5d60286516e659322583844ef63e95ba3  output/pdf/RelaySpec_End_to_End_Research_Report.pdf
333943875b74360d2c41a3489c597e07a6c8074d02538dcd352308dae5c1007b  reports/final/BREADTH_MATRIX.json
85c3125e1f3fe55bb5f1a552b256f08f98bcfbed521b286aa8d09a13a39e694b  reports/FINAL_RESULTS.md
addeca751a40415dfcabc4a5165036fd8e4225bc743063b82b758fde08cc643d  docs/plans/2026-08-28-relayspec-rigorous-evidence-and-final-benchmarks.md
d0ca15818838c2d1c6dec416bac3ef76feee1365fbfde41ccab5cd6275fa4017  reports/final/remote-4gpu-verification/allocation.jsonl
d15dc27254070db651a2cce8a7f6e95220517313e00bc7dcec202b6ad3862964  reports/final/remote-4gpu-verification/pytest.txt
```

The final evidence directory is approximately 146 MiB and contains 64
rank-separated raw benchmark JSONL files in addition to manifests, scorer
outputs, hashes, telemetry, summaries, and audits.
