# RelaySpec Clean ICLR Visual Revision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the RelaySpec manuscript as a clean anonymous ICLR-style PDF with a stronger empirical visual hierarchy and no conference-status header.

**Architecture:** Keep the official LaTeX style and scientific results unchanged. Generate publication-quality plots from the existing final result artifacts, reference those plots from the manuscript, prune visible engineering provenance, and validate both content and rendering.

**Tech Stack:** LaTeX, Python, Matplotlib, pytest, Ruff, Poppler PDF tools

---

### Task 1: Add clean-header and evidence-policy checks

**Files:**
- Modify: `tests/test_iclr_manuscript_audit.py`
- Modify: `scripts/audit_iclr_manuscript.py`

**Steps:**

1. Add a test that rejects review-status and publication-status wording in the clean PDF.
2. Add a test that rejects visible revision hashes and internal test-count claims in the manuscript.
3. Run the focused tests and confirm they fail against the current manuscript.
4. Add the minimum audit rules needed for the clean artifact.
5. Run the focused tests and confirm they pass after the manuscript revision.

### Task 2: Generate the professional result figures

**Files:**
- Modify: `scripts/build_iclr_paper_assets.py`
- Modify: `tests/test_iclr_paper_assets.py`
- Create: generated figure PDFs under `paper/iclr2027/generated/`

**Steps:**

1. Add tests for the throughput, mechanism, breadth, acceptance-margin, and memory figure outputs.
2. Build a connected-dot throughput plot from the existing MATH results.
3. Expand the mechanism plot to compare ideal and measured speedup.
4. Replace the breadth accounting panel with a task heatmap and acceptance-margin plot.
5. Add a compact EAGLE memory figure.
6. Regenerate assets and check that every plotted number traces to the final result records.

### Task 3: Restructure the manuscript

**Files:**
- Modify: `paper/iclr2027/relayspec_iclr2027.tex`

**Steps:**

1. Clear the left header immediately after `\maketitle` without editing the official style file.
2. Reorder the main empirical section around the four-question evidence hierarchy.
3. Insert the new throughput and mechanism figures.
4. Replace the current breadth visualization with the heatmap and acceptance-margin figure.
5. Keep exact main results and uncertainty in compact tables.
6. Move secondary protocol and ablation detail to the appendix.
7. Remove hashes, commits, manifest digests, numeric seeds, remote-job detail, and test-count claims from the visible PDF.
8. Correct the ethics wording so it claims unchanged target weights and verification, not universally unchanged text.

### Task 4: Build and validate the PDF

**Files:**
- Modify: `reports/ICLR_MANUSCRIPT_QA.md`
- Modify: `reports/ICLR_VISUAL_REVIEW.json`
- Create: `output/pdf/RelaySpec_ICLR_2027.pdf`

**Steps:**

1. Compile LaTeX twice and fail on unresolved references or LaTeX errors.
2. Run the focused paper tests, full test suite, Ruff, and manuscript audit.
3. Extract PDF text and verify the banned status phrases and revision hashes are absent.
4. Render every PDF page in color and grayscale.
5. Inspect all pages at readable resolution for overlap, clipping, tiny labels, and broken figure placement.
6. Update the QA and visual-review records with the final PDF digest and page count.
7. Re-run the complete audit against the final immutable PDF.
