# RelaySpec Clean ICLR Visual Design

## Goal

Produce a professional anonymous RelaySpec paper that uses the official ICLR typography and page geometry, but does not display a review or publication-status sentence. The paper should lead with the main measured result and use figures only when they make the evidence easier to judge.

## Chosen format

The manuscript keeps the unmodified ICLR 2027 style file and anonymous author block. A document-level override clears the left running header immediately after `\maketitle`. The clean reader-facing PDF therefore contains neither “Under review” nor “Published as” text. The paper remains anonymous.

The clean PDF is a reader-facing artifact. A future formal submission should be rebuilt with whatever header and ruler the conference instructions require at submission time.

## Evidence hierarchy

The main paper answers four questions in this order:

1. What computation does RelaySpec replace?
2. How much faster is it on the main MATH evaluation?
3. Why does the speedup differ across proposer and target pairs?
4. Under which workloads does the method help or hurt?

The main figures are:

1. A source-reuse versus RelaySpec system diagram.
2. A connected-dot throughput plot for autoregressive decoding, source reuse, RelaySpec, and the target-specific proposer where available.
3. A two-panel mechanism plot showing the source fraction that can be removed and the gap between ideal and measured speedup.
4. A cross-task heatmap paired with an acceptance-margin plot that visualizes the measured break-even condition.

Acceptance survival and memory results remain supporting figures in the appendix. Exact numerical values remain in compact tables where reviewers need confidence intervals, scores, or text-agreement counts.

## Detail policy

The PDF retains the scientific information needed to reproduce or judge the claims: model and proposer names, benchmark sizes, decoding protocol, hardware, precision, generation cap, relay definition, confidence intervals, official scores, acceptance, fitting cost, memory, negative cells, and limitations.

The PDF removes engineering provenance that does not help judge the result: checkpoint hashes, repository commits, manifest digests, numeric random seeds, remote job details, test counts, and repeated low-level protocol text. These records remain in the repository artifacts and machine-readable manifests.

## Visual rules

- Use one restrained color palette consistently across all figures.
- Use direct labels where possible and avoid decorative charts.
- Use the same method order and colors in every plot.
- Keep axis units explicit and show the `1.0x` no-speedup boundary.
- Do not plot accuracy as a decorative all-equal bar chart.
- Keep captions factual and short.
- Do not add revision labels, version numbers, result cards, or promotional language.

## Verification

The build must pass the manuscript audit and test suite. Every page must be rendered and inspected for clipping, overlapping labels, unreadable text, broken references, blank pages, and hidden status-header text. Text extraction must confirm that neither review-status nor publication-status wording appears in the final PDF.
