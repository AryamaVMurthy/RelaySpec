# RelaySpec contribution and comparison revision

The main argument is that a released feature-conditioned drafter can remain useful after its conditioning model changes. Small-data linear calibration makes this reuse economical while removing the original source transformer. The application, resulting deployment capability and measured findings carry the contribution. Earlier linear adapters and frozen-drafter adaptation remain credited.

## Final manuscript changes

- **Contributions, pages 1–2:** four explicit contributions cover portable inherited drafters, small-data calibration, feature-fit versus decoding behavior, and native-throughput/cross-family transfer. The former “Main insights” paragraph has been integrated here.
- **Table 1, page 2:** a direct design comparison with TriSpec, PARD and SD² identifies inherited objects, adaptation and execution/verification. It is supported by primary sources in `COMPARISON.md`. TriSpec is conceptual, PARD needs no new-target fitting for its released parallel drafter, and SD²'s default and frozen training options are distinguished.
- **Table 4, page 8:** the completed 128-question, 2,048-token-cap comparison highlights RelaySpec's highest measured throughput and own-runtime AR speedup: **193.68 tokens/s and 5.14×**, versus PARD **105.63 and 3.34×**, and frozen-drafter SD² **19.92 and 1.60×**. The runtime/model differences remain explicit beside the ranking. The numerical builder computes the bold maxima from recorded values, without changing any result or interval. `NUMERIC-COMPARISON.md` independently verifies raw request identity, counts, caps, timing and quality scope.
- **Limitations, page 9:** the ending now describes hidden-state access, checkpoint-specific calibration, preparation costs, serving conditions, greedy cross-vocabulary scope and numerical execution dependence. Experimental qualifications remain adjacent to their results.
- Repeated prose and preliminary headings were condensed while retaining **nine main pages, five main figures and four main tables**. The complete draft has **62 pages, 28 figures, 71 tables and 50 cited sources**. No new GPU experiments were run for this revision.

## Validation and artifacts

The final PDF compiles without undefined references/citations or overfull boxes. Every page is covered by color and grayscale visual review, using explicit inspection of changed pages/regions and byte identity for unchanged content. The final headline/reference adjustment changes only pages 1–2, which were reopened. The signed review is in `reports/ICLR_VISUAL_REVIEW.json`, with detailed evidence in the three `VISUAL-*.md` reports and render-comparison files in this directory.

The citation audit now includes compiled citations from input tables. A direct regression check verifies 49 citations in the main TeX source plus TriSpec in the new input table, totaling 50. `final-audit.log` records the complete manuscript and deterministic result-reproduction audit.

All **16 final manuscript checks pass**, including the nine-page limit, 50 resolved citations, raw-evidence asset reproduction, overlap checks, PDF parsing and embedded fonts, and the review tied to the final PDF hash. The audit reproduces 30 analytical visual assets and 31 autoresearch tables/raster plots from 27 isolated builders, alongside registered core and family-extension assets.

Final PDF SHA-256: `b3bca3c6bd5b5e93f743c1361c105c0083bf67741774f3ccc0bdad82c7a6a0e1`.

Reading copies:

- `output/pdf/RelaySpec-ICLR2027-main9.pdf`
- `output/pdf/RelaySpec-ICLR2027-complete.pdf`

The complete reading copy is byte-identical to `paper/iclr2027/relayspec_iclr2027.pdf`. The nine-page copy is a PDF export of main pages 1–9.
