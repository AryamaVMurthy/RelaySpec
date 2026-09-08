# Visual review: pages 10–32

Reviewed on 2026-09-08 against PDF SHA-256:

`33141db9fc8725467808e1edbe62bd190229eeb26b3377c2ed20a29d8f18ecd9`

Artifact: `paper/iclr2027/relayspec_iclr2027.pdf`.

Render coverage: all pages **10 through 32 inclusive**, in both color and grayscale, using `tmp/pdfs/rebuild/{color,gray}/page-XX.png`. Every page was visually inspected in labeled contact sheets. Additional individual-page inspection covered color pages 15, 16, 18, 19, 22, 26, 28, 30, 31 and 32, and grayscale pages 16, 17, 20, 23 and 32. The PDF hash was rechecked unchanged after inspection.

## Result

**Pass for the inspected page range.** No visible clipped text, overlapping elements, cut-off tables, missing glyphs, displaced captions, or illegible equations were found. This signoff covers visual layout and legibility, not numerical correctness, citation verification, or pages outside 10–32.

- Pages 10–14: AI/reproducibility statements and bibliography fit their margins. Long URLs wrap within the text area. The bibliography continuation onto page 12 is normal and does not obscure text. Page 14 ends early before the explicit appendix transition.
- Page 15: primary-Qwen protocol table is readable, its caption explicitly points to separate extension protocols, and long model identifiers wrap within the page.
- Pages 16–17: generation-flow arrows and labels are legible; the relocated AR and breadth plots are placed beside their supporting measurements. Figure labels, error bars, dashed references and legends remain visible in grayscale. Figure 5 distinguishes targets with different marker shapes.
- Pages 18–22: dense workload, quality and calibration tables have readable headers and aligned rows; interval brackets and mathematical symbols render correctly. The paired data/optimization figure is readable in grayscale.
- Pages 23–29: layer-depth figure, native-interface tables, and accompanying captions stay within margins. The layer-depth plot distinguishes drafter families with circle/square markers in grayscale. Table captions and subsequent paragraphs do not collide.
- Pages 30–32: the longest native-capacity and scaling tables remain legible. The capacity figure is now correctly inside the fitting/capacity study, beside its numerical table. Solid/dashed lines, circle/triangle markers, and the black dense-map star preserve interpretation in grayscale. Both panels' axis labels and legends are readable at individual-page scale.

No visual correction is required within this reviewed range. Any later PDF rebuild that changes the hash invalidates this artifact-specific signoff and requires checking affected pages again.
