# Final positioning manuscript: visual review of pages 10–35

**PASS.** No unresolved visual or scientific-scope issue was found in the reviewed page range.

Final manuscript: `paper/iclr2027/relayspec_iclr2027.pdf`.

Final SHA256: `b3bca3c6bd5b5e93f743c1361c105c0083bf67741774f3ccc0bdad82c7a6a0e1`.

The final PDF has nine main-text pages and 62 total pages. This review covers physical pages **10–35 inclusive**, not the main text or pages 36–62.

## Actual inspection and reuse

Compared every color and grayscale page image in this range against `tmp/pdfs/visuals-final-r2`, the previously reviewed build (PDF SHA256 `123ab1d05d2e8bff1a46dc68581c108ddb5060ab3fb43a19b013082425be8348`). Only **page 34** remained pixel-identical in both modes. Its prior final pass in `reports/paper-visuals-20260908/VISUAL-SCALING.md` is reused, including the corrected open/filled-marker and output-cap caption.

All other covered pages were inspected from the actual `tmp/pdfs/positioning-final` render, corresponding to candidate SHA256 `041275791761ec61508fa6bc6b0823be1aa2951606911564571ea6a43892e473`:

- **Pages 10, 11, 12, 13 and 14:** bibliography and statements, inspected in labeled contact sheets in both color and grayscale. Sheets are `tmp/pdfs/positioning-final/review-{color,gray}-10-14.png`.
- **Pages 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 and 35:** each page image opened individually in both color and grayscale. Figure/table-bearing pages were not accepted from contact sheets alone.
- **Page 34:** prior inspection reused only after independently confirming complete pixel identity in both modes.

Comparison evidence is saved in `render-comparison-10-35.json` in this report directory.

The root subsequently made two main-text changes and rendered `tmp/pdfs/positioning-final-r2`. I independently compared **all 52 images** for pages 10–35 in both modes against the candidate inspected above: every image is pixel-identical. I also independently verified the final PDF hash. Therefore this review applies to the final SHA256 above without further page reinspection. The final identity checks are saved in `render-comparison-r2-10-35.json`.

## Findings by content

| Pages | Finding |
| --- | --- |
| 10–14 | Statements and bibliography wrap within the page margins; no clipping, citation placeholders or missing text seen. |
| 15–18 | Protocol, generation cycle, primary numerical tables, token-agreement figure and workload breadth remain readable. New table numbering is consistent with the visible nearby references. Task accuracy is clearly separated from historical BF16 exact token agreement. |
| 19–20 | Block-size and historical data/optimization figures have legible legends and axes in both modes. Historical workload, cap and source-reuse controls remain explicit. The fixed-record-count study is distinguished from one continuous optimization trajectory. |
| 21–24 | Compression and precision tables fit. Own-runtime AR and full-FP32 AR are distinguished. Frozen 64-request confirmation is separated from exposed layer-selection screens. The late-layer chart preserves marker distinctions and shows request-conditional intervals. |
| 25–29 | Native compression, inherited initialization and separately frozen comparisons retain clear projection-only parameter accounting. The failed 16-record confirmation criterion is visible and is not presented as a pass. Tables and captions do not collide or run beyond margins. |
| 30–32 | The native compression figure preserves its reference/threshold lines and model/control markers in grayscale. Code-layer and SVD comparisons retain development-sample, matched-rank and workload limits. No whole-model compression or universal layer-sufficiency claim is implied. |
| 33 | Eleven distinct data sizes, fixed 8,192-update budget and all 105/128 outcomes are visible. The historical fitting table remains separate, and the nonlinear/ridge caveats remain adjacent. Footnote and bottom paragraph fit. |
| 34 | Pixel-identical to the previously reviewed final page. Capacity and loss/speed figures retain explicit 512/2,048 marker coding, the 2,048-token cap and descriptive loss/speed interpretation. |
| 35 | Capacity table and extended trajectories fit. Dashed training lines, solid validation lines and shape differences remain discernible in grayscale. Optimizer updates are correctly distinguished from record counts. Regularization prose and table reference are unclipped. |

No visual edits are requested. This review checks rendered presentation, captions, nearby interpretation and numbering; it does not rerun experiments or replace the independent raw-evidence audits.
