# Final positioning revision: visual review of pages 36–62

**PASS.** No unresolved visual or scope problem was found in the assigned
appendix range. This signoff applies to final PDF SHA-256:

`b3bca3c6bd5b5e93f743c1361c105c0083bf67741774f3ccc0bdad82c7a6a0e1`.

## Actual review and reuse

The fully reviewed predecessor is `tmp/pdfs/visuals-final-r2/{color,gray}/`,
documented in `reports/paper-visuals-20260908/VISUAL-DIAGNOSTICS.md`, including
the repaired 14B fitting figures on pages 42–43.

I independently compared every page **36–62 in both color and grayscale** with
that predecessor. Pages **60–61 are byte-identical** in both modes. Their
complete earlier review carries forward.

Pages **36–59 and 62** have tiny pixel changes, confined to table-caption numbers
and numeric cross-references. The changed pixels comprise **0.005–0.043%** of
each affected page; all other pixels, including figures, axes, tables, page
geometry and line positions, remain identical to the reviewed predecessor.
Pixel-difference bounds and the exact inspected text bands are recorded in
`tmp/pdfs/positioning-review36-62/changed-regions.json`.

I visually opened **every changed text band from each of those 25 pages**, with
surrounding full-width lines, in **both color and grayscale**. These were
displayed as seven paired image sheets:

- `changed-pages-36-39.png`
- `changed-pages-40-43.png`
- `changed-pages-44-47.png`
- `changed-pages-48-51.png`
- `changed-pages-52-55.png`
- `changed-pages-56-59.png`
- `changed-pages-62-62.png`

The sheets are under `tmp/pdfs/positioning-review36-62/`. Their crops intentionally
show only changed lines and nearby context. Unchanged regions were inherited
through pixel identity rather than described as newly viewed full pages.

## Findings

- Renumbered table labels and cross-references are legible and stay on their
  original lines in both modes. No changed glyph overlaps its neighbor, table
  rule, caption or body text.
- References in the inspected paired passages remain coherent: the 14B EAGLE
  table points to the DFlash table; rate/full-answer/budget sections point to
  their renumbered tables; composition, confirmation, memory, rollout and family
  protocols retain the intended targets.
- The enlarged 14B font exports and all other plotted curves, markers, dashes,
  captions and legends are unchanged from the reviewed predecessor. Their
  earlier color/grayscale legibility signoff therefore remains applicable.
- The public-baseline appendix continues to distinguish its own-runtime AR
  controls and capped development results. The new family/rollout discussion
  retains separate runtime, cohort, exact-match and output-cap scopes.
- Tables 38–71 in this range remain ordered; the graph sequence and figure/table
  placement are unchanged. No layout regression or new blank page appears.

## Final revision identity check

The inspected positioning candidate was rendered in
`tmp/pdfs/positioning-final/{color,gray}/`. A final tiny main-text revision then
produced `tmp/pdfs/positioning-final-r2/`.

I independently verified **byte equality for all 54 assigned page images**
(27 pages × two color modes) between these two render sets. This agrees with
`reports/paper-positioning-20260908/final-render-comparison.json`, which records
changes only on pages 1–2. I also recomputed the final PDF hash shown above.
The complete assigned-range signoff thus transfers to the final PDF by image
identity. Pages 1–35 and the numerical manuscript audit are covered separately.
