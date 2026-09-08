# Paper rebuild completion audit

8 September 2026. Objective: remake the paper from the completed results, explicitly state contributions, develop a coherent nine-page ICLR main paper, and iterate with writer/reviewer agents through verified completion.

## Requirement evidence

| Requirement | Current evidence | Result |
|---|---|---|
| Use the strongest completed results through the latest128evaluation | Main Tables1–7, full retained appendices, new generated family/rollout tables and raw-input registry | Complete |
| Explicit contribution pitch and coherent narrative | Introduction has three contributions; results proceed from primary reuse to calibration/capacity/adaptation, richer calibration and new model pairs | Complete |
| Writer and reviewer iteration | WRITER.md, WRITER-R2.md, REVIEW-R1/R2/R3.md; requested revisions checked against source | Complete |
| Nine-page main text without template distortion | Compiledaux maintextendpage9; unchanged official ICLR2027 .sty/.bst checks; fullPDF56pages | Complete |
| Required paper sections and statements | Abstract, introduction, relatedwork, preliminaries, method, experiments, discussion/limitations, conclusion, AI disclosure, reproducibility, references and appendices | Complete |
| No material contradictions in numerical/model/data scope | R3 confirms revised scope, metrics, two calibration recipes, cohort exposure, tokenbridge behavior and GPU cost accounting | Complete within reviewed evidence |
| Traceable results and benchmark identity | All primary/scaling/autoresearch/family assets regenerate; familybuilder checks rawarrays/hashes/coverage/precision/targets; rolloutbuilder checks both runs/workerconfigs/executed-source hashes | Complete |
| Related work and citation integrity |50resolvedkeys; verified closestcompetitors/source-statuslog; correctednumerical/OmniDraftmetadata and addedLlama/Numina sources | Complete within logged verification scope |
| End-to-end artifact verification | Currentmanuscript auditPASS onall17checks;7relevant evidence testsPASS; cleanLaTeXlog andgitdiffwhitespacecheck | Complete |
| Visual review | Actualrendered color+gray review coversall56pages, splitroot1–9/writer10–32/reviewer33–56; individualdense-page rechecksdocumented | Complete |
| Usable deliverables | Current .tex/.bib/generatedassets andfullPDF; ninepagemain preview andcomplete readingcopy underoutput/pdf | Complete |

Current complete-PDF SHA256:
`33141db9fc8725467808e1edbe62bd190229eeb26b3377c2ed20a29d8f18ecd9`

## Verification performed

- `make paper`: pdflatex/BibTeX compile succeeded, mainend9,56totalpages, no undefined citations/references or overfull boxes.
- `PYTHONPATH=src /home/aryamavmurthy/work/RelaySpec/.venv/bin/python scripts/audit_iclr_manuscript.py`: PASS including complete asset regeneration and PDFhash-bound visual record.
- `PYTHONPATH=src /home/aryamavmurthy/work/RelaySpec/.venv/bin/python -m pytest -q tests/test_paper_evidence.py tests/test_ar_paper_evidence.py`:7passed; only dependency deprecation warnings.
- New128aggregation directly verified all64completedlanes,128uniquequestionsperarm, tokenlengths andhashes, zeroarraydisagreements.
- Main-copy export containspages1–9; completecopy hash equalsmanuscriptPDF.

## Scope retained rather than hidden

This completes the requested manuscript rebuild using available evidence. It does not guarantee ICLRacceptance, certify untested runtime behavior, or complete every historical research-plan experiment. The paper retains BF16disagreements, narrow fitting-seed coverage, exposeddevelopmentcohorts, oneheterogeneoustokenizerpair, cappedoutputs, runtime-confoundedpublicbaselines and unresolvedtightaccuracy noninferiority. The native-parity rollout result is a separateNumina/vLLMrecipe, not a causal data-only/loss-only improvement. No newGPUexperiment was needed for this rewrite.
