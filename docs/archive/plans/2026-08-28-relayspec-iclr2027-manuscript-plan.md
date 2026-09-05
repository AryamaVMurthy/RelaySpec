# RelaySpec ICLR 2027 Manuscript Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an anonymous, nine-page ICLR 2027 paper and appendix from the completed RelaySpec evidence, then generate and verify a polished PDF.

**Architecture:** Use the official ICLR 2027 LaTeX package. Generate result macros, appendix tables, and plots from immutable JSON artifacts. Keep the main text focused on frozen proposer reuse, source-trunk removal, and the measured acceptance boundary. Put full protocol and audit material in the appendix.

**Tech Stack:** LaTeX, official ICLR 2027 style, BibTeX, Python, Matplotlib, pytest, Poppler, Ghostscript.

---

### Task 1: Install the official paper scaffold

**Files:**
- Create: `paper/iclr2027/iclr2027_conference.sty`
- Create: `paper/iclr2027/iclr2027_conference.bst`
- Create: `paper/iclr2027/README.md`

**Steps:**

1. Download the official style archive from the URL in the ICLR 2027 author guide.
2. Extract the style, bibliography, and example source in a temporary directory.
3. Copy only official reusable files into `paper/iclr2027/`.
4. Record the source URL, retrieval date, archive SHA-256, nine-page rule,
   anonymity rule, appendix rule, and AI-use statement requirement.
5. Compile the untouched example once to verify the local toolchain.

**Verify:**

```bash
sha256sum tmp/iclr-2027-style-files.zip
pdflatex -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

Expected: the official example compiles without a missing package.

### Task 2: Generate paper numbers from artifacts

**Files:**
- Create: `scripts/build_iclr_paper_assets.py`
- Create: `tests/test_iclr_paper_assets.py`
- Create: `paper/iclr2027/generated/results_macros.tex`
- Create: `paper/iclr2027/generated/breadth_table.tex`
- Create: `paper/iclr2027/generated/acceptance_table.tex`

**Steps:**

1. Write a failing test that requires four main MATH cells, sixteen breadth
   cells, correct task counts, and the recorded aggregate values.
2. Run the test and confirm that the asset builder does not yet exist.
3. Implement one reader for main summaries, the breadth matrix, memory JSON,
   adapter cost, and remote verification proof.
4. Generate named LaTeX macros so main-text numbers are never copied by hand.
5. Generate appendix tables with deterministic row order and escaped labels.
6. Run the test and confirm exact expected values.

**Verify:**

```bash
.venv/bin/pytest -q tests/test_iclr_paper_assets.py
.venv/bin/python scripts/build_iclr_paper_assets.py
```

Expected: all generated files are reproducible byte-for-byte.

### Task 3: Generate the main figures

**Files:**
- Modify: `scripts/build_iclr_paper_assets.py`
- Create: `paper/iclr2027/figures/system_overview.pdf`
- Create: `paper/iclr2027/figures/amdahl_prediction.pdf`
- Create: `paper/iclr2027/figures/acceptance_survival.pdf`

**Steps:**

1. Draw a two-panel vector system diagram that shows source reuse and
   RelaySpec using the same target and proposer.
2. Plot predicted against observed speed for all sixteen cells with a diagonal
   and labels for cells at or below one.
3. Plot accepted-prefix survival by draft position for source and relay in a
   compact appendix figure.
4. Use readable fonts, color-blind-safe colors, and vector PDF output.
5. Parse every generated PDF with Ghostscript.

**Verify:**

```bash
gs -q -dNOPAUSE -dBATCH -sDEVICE=nullpage paper/iclr2027/figures/*.pdf
```

Expected: all three vector figures parse without error.

### Task 4: Write the nine-page manuscript

**Files:**
- Create: `paper/iclr2027/relayspec_iclr2027.tex`
- Create: `paper/iclr2027/references.bib`

**Steps:**

1. Write the title, anonymous author block, abstract, and introduction around
   the approved frozen-proposer-reuse question.
2. Explain source reuse and RelaySpec in concrete inference steps.
3. Define compatible targets by the actual tokenizer, vocabulary, causal
   model-family, feature-tap, and consumed-interface requirements.
4. State the relative interface loss and acceptance-aware latency equation.
5. Explain the provider rule as a workload calibration procedure, not a free
   per-request oracle.
6. Describe models, data, partitions, metrics, timing, hardware, uncertainty,
   and frozen components.
7. Present the four main MATH results, the sixteen-cell mechanism result,
   resource evidence, and two minimal design ablations.
8. Write related work by explaining exact differences from each neighboring
   approach.
9. State limitations and the required AI-use disclosure.
10. Move full tables, survival curves, mismatch details, protocol constants,
    and reproducibility information into appendices after references.

**Verify:**

```bash
pdflatex -interaction=nonstopmode -halt-on-error relayspec_iclr2027.tex
bibtex relayspec_iclr2027
pdflatex -interaction=nonstopmode -halt-on-error relayspec_iclr2027.tex
pdflatex -interaction=nonstopmode -halt-on-error relayspec_iclr2027.tex
```

Expected: no undefined citations or references.

### Task 5: Add automated manuscript QA

**Files:**
- Create: `scripts/audit_iclr_manuscript.py`
- Create: `tests/test_iclr_manuscript_audit.py`
- Create: `reports/ICLR_MANUSCRIPT_QA.md`

**Steps:**

1. Write failing tests for anonymity, official style use, required sections,
   main-text page count, citation-key resolution, and generated-macro use.
2. Add checks for em dashes, semicolons in prose, placeholders, repeated
   paragraphs, vague banned phrases, and unsupported headline values.
3. Add a numerical cross-check between paper macros and source JSON.
4. Add the existing zero prompt-overlap result to the QA report.
5. Generate one Markdown report that lists every check and its evidence.

**Verify:**

```bash
.venv/bin/pytest -q tests/test_iclr_manuscript_audit.py
.venv/bin/python scripts/audit_iclr_manuscript.py
```

Expected: every check passes or names one exact line to fix.

### Task 6: Compile and inspect the final PDF

**Files:**
- Create: `output/pdf/RelaySpec_ICLR_2027.pdf`
- Create temporarily: `tmp/pdfs/relayspec-iclr2027/`

**Steps:**

1. Compile the full paper with BibTeX.
2. Confirm that main text ends by page nine and references start no earlier
   than needed.
3. Parse the PDF with Ghostscript and extract text with Poppler.
4. Render every page to PNG.
5. Inspect each page for clipped content, small labels, table overflow,
   misplaced floats, blank regions, and section ordering.
6. Fix every visual defect and repeat compilation and rendering.
7. Remove auxiliary build files and temporary PNGs after final verification.

**Verify:**

```bash
pdfinfo output/pdf/RelaySpec_ICLR_2027.pdf
gs -q -dNOPAUSE -dBATCH -sDEVICE=nullpage output/pdf/RelaySpec_ICLR_2027.pdf
pdftoppm -png output/pdf/RelaySpec_ICLR_2027.pdf tmp/pdfs/relayspec-iclr2027/page
```

Expected: a valid anonymous ICLR 2027 PDF with no visual defects.

### Task 7: Run independent coherence checks

**Files:**
- Modify as needed: `paper/iclr2027/relayspec_iclr2027.tex`
- Modify: `reports/ICLR_MANUSCRIPT_QA.md`

**Steps:**

1. Compare every abstract and contribution claim with the result ledger.
2. Check that every paragraph answers what is done, why it is needed, or how
   it is measured.
3. Check that terminology and symbols stay consistent from method to results.
4. Check that no appendix claim is stronger than the main-text claim.
5. Run the full repository tests, lint, protocol audit, prompt-overlap audit,
   manuscript audit, PDF parser, and visual inspection one final time.

**Verify:**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/audit_research_constants.py --protocol configs/relayspec_protocol.yaml configs/protocol_active/*.yaml
.venv/bin/python scripts/audit_iclr_manuscript.py
git diff --check
```

Expected: all checks pass and the QA report identifies no unresolved issue.

The current repository remains on `main` because the user previously required
this work to stay on the main RelaySpec tree. Edits must be limited to the new
paper, generated assets, QA scripts, tests, and their documentation. Existing
unrelated working-tree changes must not be staged or overwritten.
