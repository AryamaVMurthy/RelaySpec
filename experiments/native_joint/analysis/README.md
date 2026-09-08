# Frozen confirmation analysis

These scripts are separate from the immutable inference snapshot. They never submit GPU jobs or tune decoding from confirmation results.

- `collect_confirmation.py` reads recorded Slurm handles and copies existing outputs/metadata. It never infers termination from a polling timeout or restarts work.
- `analyze_confirmation.py` audits all64 lanes,128 unique questions,1,024 timed generations,128 AR outputs, source/model/runtime identity, and repeat determinism. Final estimates are withheld until coverage is complete.
- `quality.py` runs the existing vendored Qwen2.5-Math grader and supplied HumanEval tests in a bubblewrap sandbox. No model/mapper code is imported from RelaySpec. Dialogue has no canonical accuracy label.
- `score_collected.py` caches content-identical grading work across methods and records scorer-source hashes. A final table requires all640 method/question outputs. Logged failures need inspection before attributing score changes to a model.
- `plot_confirmation.py` draws final paired confidence intervals, workload ratios and exact-output agreement after full coverage.

Run from the repository root with `.venv/bin/python experiments/native_joint/analysis/<script>.py`. Use `collect_confirmation.py`, `analyze_confirmation.py`, `score_collected.py`, then `plot_confirmation.py`. The first two may safely run during the campaign; final comparison tables are withheld until complete coverage.

Quality-only dependencies are isolated from the model environment:

```
uv pip install --python .venv/bin/python --target /tmp/native-quality-deps antlr4-python3-runtime==4.11.1 word2number==1.1 sympy==1.12 mpmath==1.3.0
```

The sandbox binds only system runtime directories, installed Python packages, the private grading dependencies and the vendored evaluation source as read-only. The home directory and network are absent. Limits are5CPU seconds,8wall seconds,768MiB address space and1MiB output-file size. Correct/incorrect code, an infinite loop, hidden home access and a known math answer are tested by invoking `quality.py` directly. Math symbolic parsing also runs inside this sandbox.

The vendored Qwen evaluator is identified by source-file hashes. Its directory is not a separate Git checkout; the parent RelaySpec commit must not be misreported as an upstream Qwen commit. Numerical BF16 output identity, math/code task scores and throughput are distinct measurements. None alone establishes universal losslessness or task-quality equivalence.
