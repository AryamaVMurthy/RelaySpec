# Timeboxed experiment and post-results paper revision

The experiment window ended at 2026-09-06 06:26:48 UTC. The final manuscript rewrite began after that time. GPU cutoff enforcement succeeded at 06:11:48 UTC; the user queue was verified empty afterward. No additional GPU work was launched.

## Completed within the window

- EAGLE8 long-output quality: jobs 27967 and 27968 completed in 26m55s and 40m07s. Combined audit verified 128 questions, eleven methods and all 1,408 rows, including checkpoint/source/scorer identity.
- Four targeted EAGLE14 learning-rate pilots: job 27977 completed in 3m39s. Numerical equivalence, fit records and decoding checks passed.
- Proper lower-rate endpoint fits and paired endpoint decoding were not launched because their full ten-minute wall limit no longer fit before the GPU cutoff. Pilot results are not substituted for proper training evidence.

## Paper delivered

- Source: `paper/iclr2027/relayspec_iclr2027.tex`
- PDF: `paper/iclr2027/relayspec_iclr2027.pdf`
- PDF SHA256: `6213a8ee9ebf99632c7446dfb7dc09ab5a0016526a29cb3e8ccc009ddf59ea67`
- Nine pages of main text, 32 pages including policy statements, references and appendix.
- Rewritten title, abstract, introduction, positioning, controlled scaling discussion and conclusion. Controlled data and matched-capacity figures are in the main paper.
- Main text includes both 8B long-output small-data comparisons, fitting-versus-decoding counterexamples, 14B replication, regularization limits, matched warm-budget adaptation and public-runtime baselines.
- Complete EAGLE quality table added with a reproducible asset exporter. Full grids, trajectories, conservative quality intervals and scope conditions remain in the appendix.
- Replaced the stale claim–evidence map. Removed misleading implications of completed untouched confirmation, unchanged population quality or total training-compute savings.
- Removed forced appendix page breaks that isolated small amounts of content. Fixed hyperlink package ordering to eliminate duplicate destination warnings.

## Validation

`make paper` completed. The manuscript audit passed all sixteen checks, including official style, anonymity, section order, nine-page boundary, 42 resolved citation keys, raw-backed asset regeneration, overlap audits, clean LaTeX log, PDF parsing, embedded fonts, and final visual review. All 32 pages were reviewed in color and grayscale, tied to the PDF hash above. The three focused EAGLE exporter/shard-order tests passed. The Makefile export reproduced the independently generated temporary assets byte for byte.

## Explicitly unrun under the time constraint

Composition fitting/quality, full 14B long-output capacity quality, proper EAGLE14 lower-rate endpoints, generalized EAGLE matched-time adaptation, untouched confirmation and autoresearch remain unexecuted. Large-data scaling stays paused. These omissions are not reported as completed experiments. The rewritten manuscript scopes its conclusions to the completed evidence.
