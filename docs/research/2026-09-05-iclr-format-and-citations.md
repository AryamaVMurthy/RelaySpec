# ICLR format, structure and citation revision

Checked September 5, 2026. This revision rebuilds the manuscript around the
existing validated evidence. It does not introduce new experimental results.

## Format and structure

The manuscript uses the official ICLR 2027 anonymous review template, including
its review header, line numbers, typography and bibliography style. The four
bundled style files were compared byte-for-byte with the official download.
Main text ends on page 9. The complete draft has 18 pages.

The [ICLR author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
set the formatting and page requirements. They do not prescribe one universal
order of scientific headings. We use the following order:

1. Introduction
2. Related work
3. Preliminaries
4. Method
5. Experiments, including setup, results and ablation studies
6. Conclusion, including the scope of the measurements

Unnumbered AI use, ethics and reproducibility statements follow the main text.
References precede the lettered appendix. Appendix A–F covers implementation,
complete workloads, fitting studies, acceptance and numerical agreement, data
checks, and fitting time and memory.

The structure follows the recurring separation of preliminaries, method,
experimental setup, results and ablations in accepted speculative-decoding
papers. Examples reviewed include [PARD](https://arxiv.org/html/2504.18583v4),
[RepSpec at ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d70ea003729b440b89a2f958a5554c1f-Abstract-Conference.html),
and [distributed speculative inference at ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b36554b97da741b1c48c9de05c73993e-Abstract-Conference.html).
Related-work placement differs between these papers. Our early placement makes
the frozen-drafter retargeting use case clear before the method.
The earlier [OpenReview study](2026-09-05-openreview-review-study.md) remains
the review-feedback record. This revision does not claim new reviews were
retrieved for every added citation.

## Citation balance

Counts refer to distinct sources actually cited and rendered in the PDF, not
unused entries in the working bibliography.

| Category | Count | Share of 42 references |
| --- | ---: | ---: |
| Peer-reviewed conference or journal papers | 36 | 85.7% |
| Preprint-only papers and technical reports | 5 | 11.9% |
| Software | 1 | 2.4% |

The five retained preprints are the Qwen3 technical report, GSM8K, HumanEval,
MBPP and Draft-OPD. The first four document models or benchmarks actually used.
Draft-OPD is directly relevant to the distinction between fitting an interface
and updating the drafter. DeepSpec is cited separately as software.

No new preprint-only source was added. Seventeen peer-reviewed sources were
added to the cited set. Fifteen needed new bibliography entries. Existing DSI
and GRIFFIN entries are now used in the discussion. Added citations cover
feature-conditioned drafting, proposal structure, execution, self-speculation,
frozen-model adaptation, normalization and evaluation. They are attached to
specific claims rather than inserted as an unrelated citation list.

Two outdated bibliography labels were corrected: *Let's Verify Step by Step*
is an ICLR 2024 paper, and VSD is an ICML 2026 paper. Publication status is
determined from venue or institutional records, not the presence of an arXiv
URL. The HumanEval bibliography displays its first ten authors followed by
“et al.” through BibTeX's standard `and others` syntax.

The [42-source status registry](../../reports/CITATION_STATUS_2026-09-05.json)
records source URLs. The [expansion log](../../reports/citation-expansion-20260905.json)
records metadata and the purpose of each new bibliography entry. Publisher,
proceedings and author/institutional pages support the metadata and narrow
method descriptions. This is not a claim to have reproduced those papers.

## Evidence and validation

The revision rewrites the abstract and related work, moves metric definitions
before the method, and groups results and ablations under Experiments. AR
remains the primary baseline. Native throughput retention, inherited drafter
training, marginal fitting data, task quality and numerical agreement remain
separate comparisons. Historical workload and ablation results are preserved.

The manuscript uses eight figures and sixteen tables. The figure-label spacing
fix is preserved. All eighteen pages were inspected in color and grayscale.
The automated audit checks official style hashes, anonymity, page count,
citations, statements, asset consistency and the PDF-specific visual record.
The standalone source archive is compiled separately from the repository.
These checks establish artifact consistency and presentation quality, not an
acceptance prediction or resolution of the outstanding research plan.
