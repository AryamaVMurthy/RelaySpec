# Independent scientific review, round 3

Date: 8 September 2026.

Reviewed source: `paper/iclr2027/relayspec_iclr2027.tex`.

Reviewed PDF SHA-256: `33141db9fc8725467808e1edbe62bd190229eeb26b3377c2ed20a29d8f18ecd9`.

## Resolution of round 2

| Request | Current evidence | Status |
|---|---|---|
| Correct vocabulary-intersection description | The family appendix now states direct lookup for fully covered chunks, decode/re-encode fallback, possible changes to canonical token boundaries, and target verification of the resulting sequence. This agrees with `bridge_encode_new_chunk`. | Resolved |
| Remove unsupported prediction claim | The third contribution now says “Interface findings that inform deployment choices.” | Resolved |
| Restore nine-page main text | The compiled auxiliary label `maintextend` resolves to page 9. The compile log reports 56 pages total. | Resolved |
| Report resource counts for rich calibration | Main and appendix identify generation as 96.4 minutes on two GPUs, capture/validation as 52.0 minutes on two GPUs and mapper epoch computation as 14.4 minutes on one GPU. | Resolved |
| Scope answer-scoring claim | Setup now explicitly describes scoring the primary MATH-500 outputs rather than implying every new extension was independently answer-scored. | Resolved |

The revised source preserves the two operating regimes and their different objectives, data, numerical runtimes and costs. It retains the negative Llama scaling result and confines the positive cross-family gain to a recipe comparison. The rollout result is labeled as a separate Numina split with one fitted checkpoint and a repeated speculative timing comparison. Exact output agreement remains explicitly distinguished from mathematical correctness and universal numerical equivalence.

I found no remaining blocking scientific contradiction in the reviewed main narrative, new equations or extension appendices after these corrections. The inherited limitations remain real: limited fitting-seed replication, one heterogeneous pair, development exposure, capped outputs and public-runtime comparison confounds. They are disclosed and do not become resolved merely because the manuscript is coherent.

## Rendering sign-off

I visually inspected **all pages 33–56 in color and grayscale**, including individual dense pages and figures. The detailed coverage and findings are in `VISUAL-REVIEWER.md`. No blocking layout or legibility defect was found in that assigned range. Pages 1–32 are outside this reviewer's visual coverage and require the other reviewers' records. The final main boundary is verified by the compiled auxiliary file, not claimed from inspection of pages I was not assigned.

## Final reviewer disposition

**Approve the scientific presentation within the reviewed scope, conditional on the root's final whole-document artifact checks and combined visual coverage.** No new large experiment is required to make the current claims honest and internally coherent. This is not a guarantee of ICLR acceptance, universal method superiority, or closure of the stated research limitations.
