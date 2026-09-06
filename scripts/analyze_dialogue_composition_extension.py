"""Pair both fitting seeds on the fixed sixteen-conversation extension."""

import hashlib
import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907/run-28434")
configs = Path("configs/autoresearch/20260907")
inputs = {}
results = []
initial_rows = [json.loads(x) for x in Path(
    "reports/autoresearch-20260907/run-28431/lane1/benchmark-rank0.jsonl"
).read_text().splitlines()]
initial_ids = {r["problem_id"].rsplit("/turn", 1)[0] for r in initial_rows}
for interface, lanes in [("Retargeted", [0, 1]), ("Native", [2, 3])]:
    rows = []
    for lane in lanes:
        status = json.loads((root / f"lane{lane}-status.json").read_text())
        if status["status"] != "pass":
            raise ValueError("Cannot analyze incomplete extension")
        path = root / f"lane{lane}" / "benchmark-rank0.jsonl"
        shard = [json.loads(x) for x in path.read_text().splitlines()]
        manifest = json.loads((configs / f"composition-dialogue-extension-shard{lane % 2}.json").read_text())
        expected = {r["problem_id"] for r in manifest["records"]}
        observed = {r["problem_id"].rsplit("/turn", 1)[0] for r in shard}
        if observed != expected or observed & initial_ids or len(observed) != 8:
            raise ValueError("Conversation selection mismatch or initial-screen overlap")
        rows.extend(shard)
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_hashes = {}
    for method in {r["method"] for r in rows}:
        hashes = {r["mapper_checkpoint_sha256"] for r in rows if r["method"] == method}
        if len(hashes) != 1:
            raise ValueError("A checkpoint changed between shards")
        checkpoint_hashes[method] = next(iter(hashes))
    for seed in [1729, 1730]:
        result = summarize(rows, reference=f"relay_math_seed{seed}")
        if result["clusters"] != 16 or result["requests"] != 32:
            raise ValueError("Incomplete two-turn conversations")
        method = f"relay_mixed_seed{seed}"
        result["mixed_method"] = method
        result["capped_turns"] = {
            m: sum(r["output_tokens"] >= 256 for r in rows if r["method"] == m)
            for m in result["methods"]
        }
        results.append(dict(interface=interface, seed=seed,
                            checkpoint_sha256=checkpoint_hashes, **result))
        m = result["methods"][method]
        print(interface, seed, m["throughput_ratio"], m["throughput_ci95"],
              "matching outputs", m["token_matches"])

fitting = []
for job, seed in [(28430, 1729), (28433, 1730)]:
    for lane in range(4):
        path = next(Path(f"reports/autoresearch-20260907/run-{job}/lane{lane}/fitting").glob("*/validation.jsonl"))
        validation = json.loads(path.read_text().splitlines()[-1])
        if validation["step"] != 8192:
            raise ValueError("Missing fixed fitting endpoint")
        domains = validation["groups"]["validation"]["by_domain"]
        if set(domains) != {"math", "general_instruction"} or any(v["records"] != 64 for v in domains.values()):
            raise ValueError("Mismatched validation domains")
        fitting.append(dict(seed=seed, interface="Retargeted" if lane < 2 else "Native",
                            composition="math" if lane % 2 == 0 else "mixed", domains=domains))
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

Path("reports/autoresearch-20260907/dialogue-composition-extension-summary.json").write_text(
    json.dumps(dict(input_sha256=inputs, results=results, fitting=fitting,
                    scope="Sixteen additional development conversations, disjoint from the initial eight. "
                          "Both seeds use the same conversations and are not independent sample replications. "
                          "Paired bootstrap clusters both turns; conditional on fixed checkpoints. "
                          "No dialogue-quality scores; capped outputs remain included."), indent=2) + "\n"
)
