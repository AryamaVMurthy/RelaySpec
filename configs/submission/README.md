# Submission execution ledger

`plan.json` tracks the research tasks in the [execution plan](../../docs/plans/2026-09-05-relayspec-evidence-execution-plan.md). It is a planning ledger, **not a runnable training configuration or a registry of validated metrics**. Historical experiment configurations are preserved separately.

```bash
make research-plan       # validate organization; show next work; no GPU jobs
make submission-ready    # additionally require completed required tasks
```

All tasks initially remain `planned`. E13 (forecast) and E15 (drift) are optional for the static paper. E17 must remove unsupported optional claims if those tasks are not completed. Required tasks represent the strong target version; changing that scope requires an explicit dated amendment to the ledger and prose plan, not quietly marking an unfinished task optional.

When work is actually completed, update its status and add the named completion record under `reports/submission-execution/`. Example record structure:

```json
{
  "task_id": "E01",
  "summary": "Describe the actual result, checks, remaining limitations and interpretation.",
  "evidence": [
    {
      "path": "reports/submission-execution/ACTUAL_REPORT.json",
      "sha256": "ACTUAL_SHA256_OF_THE_REPORT"
    }
  ]
}
```

These placeholders are not completion evidence. Use real report paths and hashes. A valid unfavorable result can complete an experiment; meeting a preferred numerical outcome is not a completion requirement. Code and statistical correctness still need review: the checker verifies dependency order, issue coverage, local files and evidence hashes, not the truth of an uploaded report.

New amended GPU configs will live under `configs/submission/runs/` after the protocol and correctness repairs. They are not created or launched by the planning checker. The existing experiment entry points and implementation tasks are documented in the execution plan.
