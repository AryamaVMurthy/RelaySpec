# Main-text visual review

Root reviewed every main page (1–9) of the 62-page draft in `tmp/pdfs/positioning-final/` at 100 dpi. Color pages 1–9 were opened individually. Grayscale pages 3, 5, 7, 8 and 9 were opened individually; grayscale pages 1, 2, 4 and 6 were byte-identical to their inspected color counterparts.

The final rebuild changes only pages 1–2 in both render modes. Root reopened those final color pages in `tmp/pdfs/positioning-final-r2/` and verified that their grayscale counterparts are byte-identical. Pages 3–62 are byte-identical between the two candidates, as recorded in `final-render-comparison.json`.

Final PDF SHA-256: `b3bca3c6bd5b5e93f743c1361c105c0083bf67741774f3ccc0bdad82c7a6a0e1`.

All nine main pages pass for clipping, overlap, readable text and equations, figure legends, caption placement, table widths, and float order. The new conceptual comparison is Table 1 on page 2, below its first discussion. The public timing comparison is Table 4 on page 8, with the highest measured throughput and own-runtime AR speedup in bold. Its runtime qualification is adjacent. Five main figures remain readable in color and grayscale through marker shapes and line styles. Contributions span pages 1–2, Limitations and Conclusion remain on page 9, and the policy statements start on page 10. No font shrinking, margin override, or negative-spacing workaround was introduced.

# Scientific positioning review

An independent reviewer checked Contributions, Related work, the conceptual table, the measured public-runtime claim, and Limitations against the source and numerical reviews. No material accuracy issue was found. The suggested stronger fourth contribution headline, “Frozen reuse reaches native throughput and crosses model families,” was adopted. The result paragraph continues to state the unresolved Llama gain and the measured cross-family improvement.

The application of small-data linear calibration is explicitly part of the contribution. The manuscript credits earlier linear/frozen-adapter constructions and distinguishes empirical portability with source removal from a claim to have invented linear mapping. The existing benchmark evidence supports the stated ranking of three tested configurations. It does not support an unqualified ranking of all competing algorithms.
