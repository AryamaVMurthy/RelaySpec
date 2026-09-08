# Reviewer visual inspection

Date: 8 September 2026.

PDF: `paper/iclr2027/relayspec_iclr2027.pdf`

PDF SHA-256: `33141db9fc8725467808e1edbe62bd190229eeb26b3377c2ed20a29d8f18ecd9`

Assigned and inspected coverage: **pages 33–56 inclusive, in color and grayscale**. This is a 24-page subset of the 56-page PDF. This reviewer does not claim visual inspection of pages 1–32; those are assigned to the root and writer reviewers.

## Inspection method

I actually opened and visually inspected all six four-page contact sheets for each rendering mode, generated from `tmp/pdfs/rebuild/{color,gray}/page-XX.png`. The contact sheets are `tmp/pdfs/rebuild/reviewer-{color,gray}-{33,37,41,45,49,53}.png`. Each underlying page render is 850 by 1,100 pixels. Inspection considered layout, margins, clipping, collisions, table/caption placement, equation visibility and figure interpretability.

I additionally opened individual page images at their supplied native resolution:

- Color: **37, 43, 53, 54, 55, 56**. These cover dense capacity tables, the public-method comparison, the new rollout equation, the full extension method/protocol and the final figure/table.
- Grayscale: **34, 39, 40, 48, 52, 56**. These cover the six-panel regularization plot, dense fitting-trajectory plots, composition markers, memory bars and the final training-context bars.

## Results

| Coverage | Color | Grayscale | Findings |
|---|---|---|---|
| 33–36 | Pass | Pass | Fitting curves and regularization panels are contained, captions and tables remain legible. |
| 37–40 | Pass | Pass | Dense numerical tables fit within margins. The 14B trajectory figures are dense but use line/marker distinctions in grayscale. |
| 41–44 | Pass | Pass | Adaptation and competitor tables fit. Wrapped prose cells in the method table do not collide. |
| 45–48 | Pass | Pass | Acceptance equation and survival curves are legible. Composition groups retain distinct markers without color. |
| 49–52 | Pass | Pass | Composition/confirmation tables are readable. Memory bars retain labels, values and distinguishable gray levels. |
| 53–56 | Pass | Pass | New rollout and family appendix text, normalization equation, final figure and training-context table are contained and readable. |

No clipping, text/figure overlap, hidden hyperlink boxes, cut-off mathematical terms or layout defect requiring correction was found in this assigned range. Some appendix tables and multi-curve figures are dense; the inspected native-resolution images and accompanying numerical tables remain usable. The sign-off concerns the stated rendering, not publication acceptance or independent verification of every scientific result.

The PDF hash was checked before and after inspection and was unchanged. Any later PDF revision requires confirming the changed pages against this record.
