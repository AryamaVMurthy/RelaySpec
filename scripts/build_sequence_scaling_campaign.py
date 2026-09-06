"""Build nested natural-text distractor contexts with an identical question suffix.

This is a controlled context-cost study, not a standard long-context quality suite.
"""

import copy
import hashlib
import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from relayspec.benchmarking import benchmark_turns
from relayspec.sequence_scaling import token_ids_digest, validate_exact_input

ROOT = Path("configs/submission/competitors-length-20260906")
REV = "b968826d9c46dd6066d109eabc6255188de91218"
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-8B", revision=REV, local_files_only=True
)
source = Path("configs/eval_manifest.json")
records = json.loads(source.read_text())["records"]
questions = [r for r in records if r["benchmark"] == "math500"][:16]
texts = [r["prompt"] for r in records if r["benchmark"] == "math500"][16:]
context = tok.encode("\n\n".join(texts), add_special_tokens=False)
assert len(context) > 32768
prefix = tok.encode(
    "<|im_start|>user\nThe following are background notes. Answer only the final question.\n\n",
    add_special_tokens=False,
)
for length in [4096, 8192, 16384, 32768]:
    rows = []
    for i, q in enumerate(questions):
        suffix = tok.encode(
            "\n\nFinal question:\n"
            + benchmark_turns(q)[0]
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        count = length - len(prefix) - len(suffix)
        # Rotate real problem statements for each independent context family.
        offset = i * 113
        body = (context[offset:] + context[:offset])[:count]
        ids = prefix + body + suffix
        r = {
            "benchmark": "context_scaling",
            "problem_id": f"context-{i:02d}-n{length}",
            "prompt": q["prompt"],
            "answer": q.get("answer"),
            "context_tokens": length,
            "base_problem_id": q["problem_id"],
            "input_ids": ids,
            "input_ids_sha256": token_ids_digest(ids),
        }
        validate_exact_input(
            r, vocab_size=len(tok), max_positions=40960, output_cap=256
        )
        rows.append(r)
    manifest = {
        "records": rows,
        "tokenizer_revision": REV,
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Controlled distractor-context cost, exact post-template token count, fixed final question across lengths. Not a validated long-context quality benchmark.",
    }
    (ROOT / f"context-{length}.json").write_text(json.dumps(manifest) + "\n")
    for family in ["dflash", "eagle3"]:
        base = Path(
            f"configs/submission/confirmation-gsm8k-20260906/frozen-dense-v1/{family}-full.yaml"
        )
        c = yaml.safe_load(base.read_text())
        c["generation"]["max_new_tokens"] = 256
        c["benchmark"].update(
            manifest_path=str(ROOT / f"context-{length}.json"),
            benchmarks=["context_scaling"],
            max_prompts=16,
        )
        c["benchmark"]["methods"] = [
            m for m in c["benchmark"]["methods"] if "n2048" not in m
        ]
        c["relay_probe"]["variants"] = {
            k: v for k, v in c["relay_probe"]["variants"].items() if "n2048" not in k
        }
        c["run_name"] = f"{family}-context-{length}"
        (ROOT / f"{family}-context-{length}.yaml").write_text(
            yaml.safe_dump(c, sort_keys=False)
        )
        if length == 32768:
            pilot = copy.deepcopy(c)
            pilot["benchmark"]["max_prompts"] = 4
            pilot["generation"]["max_new_tokens"] = 32
            (ROOT / f"{family}-context-pilot.yaml").write_text(
                yaml.safe_dump(pilot, sort_keys=False)
            )
print("Built 4 context lengths, 16 requests each, DFlash and EAGLE families.")
