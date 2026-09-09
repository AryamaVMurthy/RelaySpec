"""NVTX/CUDA profiler ranges for separate, warmed hardware diagnostic runs."""
from contextlib import contextmanager
import json


@contextmanager
def annotated_forwards(target, native, compact):
    import torch
    saved = []
    counter = {"target": 0}
    def wrap(model, label):
        original = model.forward
        saved.append((model, original))
        def forward(*args, **kwargs):
            stage = label
            if label == "target":
                stage = "target_prefill" if counter["target"] == 0 else "target_decode"
                counter["target"] += 1
            with torch.cuda.nvtx.range(stage):
                return original(*args, **kwargs)
        model.forward = forward
    try:
        wrap(target, "target")
        wrap(native, "draft")
        wrap(compact, "draft")
        yield counter
    finally:
        for model, original in saved:
            model.forward = original


def profile_pass(args, spec, records, encoded, generate, target, native, compact):
    import torch
    methods = ["native", "compact_linear", "candidate", "reference"] if args.profile_method == "all" else [args.profile_method]
    record = records[0]
    ids = encoded[record["problem_id"]]
    rows = []
    # Validate that the annotation wrappers themselves preserve each output.
    expected = {}
    for method in methods:
        r = generate(method, ids, spec["output_cap"])
        expected[method] = (r.output_ids.clone(), list(r.acceptance_lengths))
        del r
    with annotated_forwards(target, native, compact) as counter:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:
            for method in methods:
                counter["target"] = 0
                with torch.cuda.nvtx.range(method):
                    result = generate(method, ids, spec["output_cap"])
                    torch.cuda.synchronize()
                wanted, acceptance = expected[method]
                assert torch.equal(result.output_ids, wanted) and result.acceptance_lengths == acceptance
                rows.append({"method": method, "problem_id": record["problem_id"],
                             "output_tokens": result.output_ids.shape[1] - ids.shape[1],
                             "target_calls": counter["target"], "annotation_identity": "pass"})
                del result
        finally:
            torch.cuda.profiler.stop()
    result = {"status": "pass", "kind": "instrumented_hardware_profile", "rows": rows,
              "scope": "One warmed request. Profile durations are diagnostics, not benchmark TPS. Hardware counters cover only kernels selected by the external profiler."}
    (args.output / "profile.json").write_text(json.dumps(result, indent=2))
    return result
