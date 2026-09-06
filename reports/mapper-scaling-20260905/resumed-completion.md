# Completion of both resumed studies

The user authorized fresh held-out confirmation and math-only versus mixed-domain mapper experiments, removing the previous 90-minute total limit. Both are complete. All ten resumed jobs finished successfully, and the Turing user queue was empty at the final scheduler check. Four GPUs were allocated per job; jobs ran sequentially. Large distinct-data scaling was not resumed.

## Job inventory

| Job | Experiment | Wall time |
|---|---|---|
| 28132 | Exact math-cache pilot | 2m54s |
| 28133 | Exact mixed-cache pilot | 2m44s |
| 28134 | DFlash confirmation pilot | 1m16s |
| 28135 | EAGLE-3 confirmation pilot | 1m22s |
| 28137 | Four math-only mapper fits, 8,192 updates | 4m36s |
| 28138 | Four mixed-domain mapper fits, 8,192 updates | 4m21s |
| 28141 | DFlash frozen confirmation, 256 × 5 methods | 18m50s |
| 28142 | EAGLE-3 frozen confirmation, 256 × 5 methods | 23m50s |
| 28145 | Composition decoding pilot, 110 outputs | 2m58s |
| 28152 | Composition decoding, 1,760 outputs | 52m30s |

All operational pilots completed below ten minutes. Full mapper training took under five minutes per arm. Full answer generation dominates evaluation wall time. Cache extraction was previously completed and separately measured at 111.4 and 107.1 seconds; it is not included in the fitting allocation times above.

## Frozen confirmation

Existing seed-1729 dense maps trained on 512 and 2,048 examples were frozen before generation. Each drafter family evaluated the same 256 GSM8K questions excluded from recorded development history, together with AR/native/source controls. The 256-question reserve remained unused. Local history exclusion does not establish absence of pretraining exposure or semantic similarity.

- DFlash N512 retains **97.48% [96.90,98.06]%** of N2048 throughput, at 157.77 versus 161.85 tokens/s.
- EAGLE-3 N512 retains **98.90% [98.14,99.66]%**, at 100.03 versus 101.14 tokens/s.
- Both pass the predeclared speed criterion. DFlash has 236/256 correct for both maps and AR; EAGLE maps have 238/256 versus AR's 236/256.
- Conservative accuracy-difference intervals versus AR are **[−3.07,3.07] pp** and **[−1.63,3.14] pp**. Neither establishes the predeclared one-point accuracy noninferiority margin; the joint criterion is not established.
- These are fixed-checkpoint paired-request intervals, not estimates of fitting-seed variation. Reusing the same 256 questions across families is not an independent task replication.

Authoritative evidence: `resumed-confirmation/{dflash,eagle3}-full-audit.json` and immutable protocol/ledgers.

## Training composition

Each arm uses 2,048 distinct records and 8,192 batch-four updates: Numina math only, or 1,024 math plus 1,024 Dolly general instructions. Dense, linear-512, MLP-512 and a second dense seed give eight fits. Identical cached validation features cover 1,024 math and 1,024 general records. Records and updates are matched; non-padding tokens are not (385,492 versus 316,518).

Mixed dense fitting lowers general-domain validation error by **32.9%** and raises math error by **10.2%**. The second seed reproduces the tradeoff. Linear-512 has lower validation error and faster decoding than matched MLP-512 in both arms.

The fixed downstream campaign evaluates 32 initial inputs each from MATH, GSM8K, HumanEval and MT-Bench. Two-turn conversations expand this to 160 timed requests, times 11 methods. All eight endpoints remain included. Generated outputs are capped at 2,048 tokens; full raw method/request/checkpoint identity and pinned scoring replay passed.

| Mapper | MATH mixed/math | GSM8K | HumanEval | MT-Bench |
|---|---:|---:|---:|---:|
| Dense | 0.988 | 1.016 | 1.029 | 1.049 |
| Linear-512 | 0.938 | 0.974 | 1.039 | 1.116 |
| MLP-512 | 0.922 | 0.959 | 1.036 | 1.108 |
| Dense, second seed | 0.983 | 1.012 | 1.024 | 1.047 |

All matched math/mixed pairs produced identical generated token sequences on all 160 requests. Gains therefore do not result from shorter answers. All eight maps scored 22/32 MATH and 30/32 GSM8K; AR scored 22/32 and 31/32. Four MATH outputs per method reached the cap, with none elsewhere. Between-arm conservative accuracy intervals are ±12.80 pp: equal observed results do not establish tight population quality noninferiority. Code and chat quality were not scored. Paired intervals and all cap counts are in the paper; conversations are resampled as units.

The result strengthens a workload-dependent story: general instructions improve general-domain feature fitting and code/chat speed, with math tradeoffs that are larger for smaller maps. It does not establish a universal best training mixture.

Authoritative evidence: `resumed-composition/{math,mixed}-fits-audit.json`, `common-validation-identity.json`, and `resumed-composition-evaluation/full-audit.json`.

## Paper and scope

The main paper includes frozen small-data confirmation and the composition finding. The appendix includes all eight fits, sixteen paired task comparisons, quality/cap counts, uncertainty, timing and protocol limitations. Tables and the new composition graph regenerate from audited artifacts using `scripts/build_resumed_paper_assets.py`.

The historical 90-minute completion inventory remains historical. Full-answer 14B capacity quality, proper lower-rate EAGLE14 endpoints and broader adaptation/autoresearch are not completed by this two-study task. They are not silently presented as completed, and no further experiments were launched.

Final verification: all 16 manuscript checks passed, including generated-asset reproduction, anonymous formatting, citation resolution, clean LaTeX output and the nine-page main-text limit. The 36-page PDF was visually reviewed in color and grayscale. The historical EAGLE export reproduces after restoring its legacy scope text, and the two focused export tests passed. Raw collected runs remain preserved locally under the resumed run directories and remotely in their immutable experiment projects.
