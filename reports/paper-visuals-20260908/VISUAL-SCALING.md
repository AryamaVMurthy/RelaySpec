# Independent rendered-paper review: pages 10–35

Candidate reviewed: `paper/iclr2027/relayspec_iclr2027.pdf`

SHA256: `a1a670f9b661fb51cb75dd90ceec3724f5b33f7206abdc85ae20f530728a184d`

PDF length: 62 pages. This review covers physical pages **10 through 35 inclusive**; it does not claim inspection of pages 1–9 or 36–62.

## Inspection method

- Opened actual color and grayscale rendered images from `tmp/pdfs/visuals-final/{color,gray}/page-XX.png` for every covered page.
- Plain bibliography/prose pages 10–14 were inspected in labeled contact sheets in both render modes.
- Every table/figure-bearing page **15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35** was additionally opened individually in both color and grayscale.
- Read the captions and nearby scientific qualifications against the displayed figures/tables. Checked label clipping, collisions, table width, small type, marker distinctions, caption placement, numbering and float order.
- Contact sheets are saved as `tmp/pdfs/visuals-final/review-{color,gray}-{10,16,22,28,34}.png` for audit convenience. Individual images, rather than contact sheets alone, were used for dense pages.

## Result

**No visual clipping, table overflow, obscured data, broken float order or unreadable figure legend found on pages 10–35.** The earlier block-figure legend issue is fixed: added vertical margin on page 19 places the legend clear of all data and confidence intervals. The shortened progress label fits. Main numerical rows remain available in adjacent tables where compact chart labels abbreviate them.

One minor caption-completeness correction remains in this candidate: **Figure 14 on page 34 does not explain open versus filled markers or explicitly state the output cap.** Open markers use 512 records; filled markers use 2,048; outputs have a 2,048-token cap. The root has been asked to add this information. Its existing continuation-arrow explanation and distinction between training loss, validation loss and decoding performance are correct. This is a caption omission, not a numerical or graphical error.

## Page-level observations

| Pages | Content and result |
|---|---|
| 10–14 | Bibliography and reproducibility prose: line wrapping stays inside margins; no unresolved citation placeholders seen. |
| 15–16 | Protocol, generation diagram and numerical main-result tables: captions fit; target verification/cropping ownership and end-to-end timing are explicit. |
| 17 | Historical BF16 accuracy versus exact token agreement: the outcomes are clearly distinguished, counts/percent axes are compatible, and the caption separates later precision-controlled extensions. The breadth chart remains legible in grayscale through circle/square markers. |
| 18 | Complete breadth tables: historical and current runs retain their own denominators; no cross-hardware normalization is implied. Dense rows remain readable. |
| 19 | Code quality and block-size figure: paired task counts and reused expanded tests are qualified; naive source reuse and the 64-request, 512-token historical block screen are explicit. Legend fix passes. |
| 20 | Historical data/optimization figure: fixed distinct data and one continuous fit are separated; this is explicitly a different dataset/budget from the new main data curve. |
| 21–22 | Smaller-verifier and precision diagnostics: own-runtime AR versus FP32 AR is distinguished. Exact token agreement is not conflated with task correctness or a global guarantee. Tables and captions fit. |
| 23–24 | Frozen two-layer confirmation and equal-capacity layer selection: exposed development screens are separated from the 64-question confirmation. Error intervals and 128-token early/spaced/late screen limits are clear. |
| 25–29 | Native-interface compression/initialization: released/repacked controls, trainable projection counts, inherited initialization and independently frozen request sets are distinguished. The failed 16-record confirmation criterion is reported rather than hidden. No misleading global parameter-reduction claim was found. |
| 30 | Native compression graph: all five arms per family are shown; the 95% threshold, released-native denominator and separate 64-question sets are clear. Grayscale preserves circle/square distinctions and visible uncertainty. |
| 31–32 | Code calibration and activation-aware SVD tables: reused development requests, rank/compute limitations and exact task-pass counts are explicit; no confirmatory claim is inferred from these screens. |
| 33 | Expanded data table: eleven distinct record counts, one fixed 8,192-update budget and 105/128 observed correctness are internally consistent. Historical fitting choices remain in a separate table. |
| 34 | Full capacity fit and loss/speed diagnostic: both are legible in grayscale, with shape/line-style distinctions. The caption accurately states that lower training loss can coexist with slower decoding and that validation loss generally associates with speed. Add marker-record/cap explanation noted above. |
| 35 | Capacity table and longer fitting trajectories: the text distinguishes optimization, training versus validation loss and fitting-only diagnostics; axes are optimizer updates rather than distinct record counts. All tick labels and legend entries fit. |

This is a rendered presentation and scientific-scope review, not an independent rerun of every experiment. The separate source-hash and numerical manuscript audits remain the evidence for raw experiment fidelity.

## Final rebuilt PDF: pass

Final PDF SHA256: `123ab1d05d2e8bff1a46dc68581c108ddb5060ab3fb43a19b013082425be8348` (verified directly from the PDF). The final file still has 62 total pages and nine main-text pages.

Independently compared the reviewed and final color/grayscale images with pixel differences for every page in **10–35**. Page **34** is the only changed page in this review range; pages **10–33 and 35 are pixel-identical** in both render modes. This agrees with `final-render-comparison.json`.

Opened final page 34 individually in both color and grayscale at `tmp/pdfs/visuals-final-r2/{color,gray}/page-34.png`. Figure 14 now explicitly states the **2,048-token cap** and **open/filled markers = 512/2,048 records**, while retaining the optimizer-update arrow explanation and bounded interpretation of training/validation loss. The revised caption fits without clipping or overlap. Both figures, legends and axes remain legible.

**Final result: PASS for pages 10–35, with no unresolved visual or caption-scope findings in this review range.** Earlier candidate observations are retained above as an audit trail; their sole outstanding caption correction is resolved in this final PDF.
