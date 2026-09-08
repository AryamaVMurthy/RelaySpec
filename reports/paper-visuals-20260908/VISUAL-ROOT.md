# Main-paper visual review

Reviewed PDF SHA256 `a1a670f9b661fb51cb75dd90ceec3724f5b33f7206abdc85ae20f530728a184d`, 62 pages, main-text end on page 9.

Root opened every page 1–9 individually from the final 100-dpi color render. Opened grayscale pages 3, 5, 7, 8, 9 individually; grayscale pages 1, 2, 4, 6 are pixel-identical to their already inspected color renders after RGB conversion.

- All five main figures fit inside margins, with visible axes, captions, legends and uncertainty intervals.
- Marker shapes and line styles preserve method identity in grayscale. Workflow labels and feedback arrow remain legible.
- Source/relay/native comparisons use matched references, with no absent native DFlash14B point invented.
- The main capacity panel uses separate campaign AR controls and distinguishes 512/2048 record counts.
- Adaptation graph correctly says end-to-end throughput. Earlier manuscript caption called it decoding throughput; this was corrected after tracing the common request-time aggregation.
- Transfer panels visibly identify different numerical runtimes, cohorts and reference denominators. The unresolved within-family gain remains in the prose.
- Main quality and heterogeneous public-runtime tables remain visible. Four contributions remain explicit across pages 1–2.
- Figure order is 1–5, with the scaling plot floated to the top of page 7 after its introduction on page 6. All main findings and the conclusion end by page 9. AI-use statement follows the main-text marker.
- No clipped text, overlapping chart elements, obscured data, visible hyperlink boxes, overfull boxes or illegible main tables/equations found.

Appendix pages are independently reviewed in VISUAL-SCALING.md and VISUAL-DIAGNOSTICS.md. This record covers only pages 1–9.

## Final rebuild

Final PDF SHA256 `123ab1d05d2e8bff1a46dc68581c108ddb5060ab3fb43a19b013082425be8348` remains nine main pages and62total. All nine main pages are pixel-identical in both color and grayscale to the individually reviewed candidate. Only appendix pages34,42,43 changed, as recorded in `final-render-comparison.json`. Main-page visual pass therefore applies unchanged.
