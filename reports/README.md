# Research evidence

Start with [the September 5 review](ICLR_SUBMISSION_REVIEW_2026-09-05.md) and
[repair plan](../docs/plans/2026-09-05-iclr-submission-repair-plan.md). They
supersede earlier “complete” or “final” status statements.

| Location | Contents |
|---|---|
| `final/MAIN_PAIRED_AR.json` | Latest main paired request-time summaries |
| `final/TRANSFER_PAIRED_AR.json` | Latest transfer summaries |
| `final/` | Selected evidence, breadth/memory results, and audits |
| `design-selection/` | Development experiments supporting interface choices |
| `training/` | Recorded fitting runs |
| `submission-review-2026-09-05/` | Small diagnostic checks and review hashes |
| Other experiment directories | Historical measurements and provenance |

“Final” is a historical directory name, not current scientific approval.
Latest summaries still need a complete archive of their remote raw records.

Source snapshots are immutable records of code used for each run. Repeated
files preserve each run's provenance. Git stores identical contents once
internally; deleting snapshot paths would lose these associations.

`COMPLETION_AUDIT.md`, `FINAL_RESULTS.md`, `EXECUTION_STATUS.md`,
`ICLR_MANUSCRIPT_QA.md`, and `claim-evidence-map.md` are earlier reports.
Consult their dates and the current review before relying on their claims.
Visual-review approval applies only to its recorded PDF hash.
