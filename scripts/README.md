# Command guide

Run commands from the repository root after installing the package.

| Task | Entry points |
|---|---|
| Fit a relay | `train_relay.py` |
| Benchmark | `benchmark_relay.py`, `benchmark_eagle3.py` |
| Prepare data | `build_eval_manifests.py`, `build_evalplus_mbpp_manifest.py` |
| Aggregate/score | `aggregate_results.py`, `merge_benchmark_runs.py`, `score_mbpp.py` |
| Analyze | `analyze_profile.py`, `build_breadth_matrix.py`, `build_breadth_paired_ar.py` |
| Build paper assets | `build_iclr_paper_assets.py`, `build_transfer_figures.py` |
| Audit paper | `audit_iclr_manuscript.py` |
| Audit overlap | `check_prompt_overlap.py`, `audit_prompt_similarity.py` |

Experimental tools include `fit_relay_closed_form.py`, `lora_sft_drift.py`,
`analyze_relay_manifold.py`, and `diagnose_cross_family_divergence.py`.
The drift exporter and the interpretation of the closed-form comparison have
known issues in the submission review. Their presence does not indicate validation.

`archive/` contains superseded document builders. Use `make paper` and
`make paper-assets` for the current manuscript.
