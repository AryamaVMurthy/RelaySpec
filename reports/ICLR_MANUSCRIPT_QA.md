# RelaySpec ICLR 2027 manuscript QA

Generated 2026-08-28 from the compiled anonymous manuscript and final result artifacts.

- Overall: **PASS**
- Main-text boundary: page 9 of the allowed 9
- Complete PDF: 17 pages including references and appendix
- Resolved citation keys: 24

| Check | Status | Evidence |
|---|---|---|
| official ICLR 2027 style | PASS | official style and bibliography files match recorded hashes |
| anonymous review source | PASS | review mode is active and no local identity marker appears in source or PDF |
| clean anonymous status header | PASS | the official anonymous layout is retained without a review or publication-status sentence |
| reader-facing scientific detail | PASS | the manuscript omits hashes, numeric seeds, internal test counts, and runtime provenance records |
| nine-page main-text limit | PASS | main-text boundary is on page 9 |
| required sections and order | PASS | all main sections and policy statements are present before references and appendix |
| citation resolution | PASS | 24 unique citation keys resolve |
| generated result assets | PASS | all generated tables, macros, and figures match final JSON artifacts |
| prompt separation audits | PASS | exact-overlap and unique-token similarity artifacts reproduce from the fixed manifests |
| language constraints | PASS | no semicolon, em dash, banned phrase, or placeholder appears in manuscript source |
| paragraph overlap | PASS | no exact or at least 0.90 Jaccard duplicate among paragraphs of 35 or more words |
| fit and evaluation exact-overlap audit | PASS | 0 normalized exact matches across 4,096 fit and 1,250 evaluation records |
| compiled LaTeX log | PASS | no undefined citation, undefined reference, or overfull box |
| PDF parse and page format | PASS | 17 pages, US Letter, unencrypted, and identity scan clean |
| embedded fonts and PDF parser | PASS | 24 font records are embedded and Ghostscript parses every page |
| recorded visual review | PASS | manual color and grayscale review covers all 17 rendered pages |

## Claim-scope checks

- The 16-cell latency result is called same-run accounting, not independent prediction.
- The workload choice is called descriptive because its speed interval uses the measured cell.
- Greedy correctness is separated from BF16 exact text agreement.
- EAGLE text mismatches and the absence of an MT-Bench judge are stated.
- Compatibility is limited to the tested Qwen3 setting.
- The fit sequences are disclosed as containing MATH training solutions.
- The paper states zero normalized exact matches and separately discloses close MATH templates.

## Visual review

The review record is tied to the compiled PDF SHA-256. It covers every page in color and grayscale. It checks clipping, overlap, table order, equation legibility, figure legibility, and hidden link boxes.
