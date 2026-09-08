"""Freeze code/instruction rollout prompts from pinned raw calibration manifests."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join(text.lower().split())


def grams(text):
    tokens = re.findall(r"\w+|[^\w\s]", normalized(text))
    return {tuple(tokens[i:i+5]) for i in range(len(tokens)-4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    exact, sets, inverted = set(), [], defaultdict(set)
    def add(text):
        exact.add(hashlib.sha256(normalized(text).encode()).hexdigest())
        value = grams(text)
        if len(value) >= 10:
            index = len(sets); sets.append(value)
            for gram in value:
                inverted[gram].add(index)
    references = json.loads(Path("data/development.json").read_text())["records"]
    for row in references:
        add(row.get("prompt") or row["turns"][0])
    old = json.loads(Path("data/progress-native.json").read_text())
    exact.update(r["question_sha"] for e in old["entries"] for r in e["records"])
    sources = [("code", args.source_root/"scaling/code-calibration-v1", "train-512.json"), ("instruction", args.source_root/"composition/math-dolly-v2-rebuild", "train-mixed-2048.json")]
    selected, provenance, rejected = [], [], []
    for domain, directory, train_file in sources:
        gate = json.loads((directory/"manifest-gate.json").read_text())
        assert gate["status"] == "pass"
        for split, file, count in [("train", train_file, 64), ("validation", "validation.json", 16)]:
            source = directory/file
            assert digest(source) == gate["files"][file]["sha256"]
            provenance.append({"domain": domain, "split": split, "path": str(source), "sha256": digest(source), "source_gate_sha256": digest(directory/"manifest-gate.json")})
            count_before = len(selected)
            for row in json.loads(source.read_text())["records"]:
                if domain == "instruction" and row["domain"] != "general_instruction":
                    continue
                prompt = row["problem"]
                key = hashlib.sha256(normalized(prompt).encode()).hexdigest()
                assert key == row["normalized_problem_sha256"]
                value = grams(prompt)
                candidates = set().union(*(inverted[g] for g in value)) if len(value) >= 10 else set()
                reason = "exact" if key in exact else ("near" if any(len(value&sets[i])/len(value|sets[i]) >= .6 for i in candidates) else None)
                if reason:
                    rejected.append({"question_sha": key, "reason": reason, "domain": domain, "split": split})
                    continue
                selected.append({"split": split, "domain": domain, "problem": prompt, "question_sha": key, "source_row": row["source_row"]})
                add(prompt)
                if len(selected)-count_before == count:
                    break
            assert len(selected)-count_before == count
    result = {"status": "pass", "records": selected, "sources": provenance, "rejected": rejected, "evaluation_reference_sha256": digest(Path("data/development.json")), "scope": "Native rollouts use prompts only. Existing raw CodeAlpaca and Dolly calibration manifests are re-filtered against the standalone evaluation manifest and each other; normalized equality and five-shingle Jaccard>=0.6 exclusion (fewer than10 shingles use exact only). Also exclude exact old Numina calibration question hashes. No semantic or pretraining-independence guarantee."}
    Path("data/progress-prompts.json").write_text(json.dumps(result, indent=2)+"\n")
    Path("data/PROGRESS_PROMPT_ATTRIBUTION.md").write_text("# Calibration prompt attribution\n\n"+(sources[0][1]/"ATTRIBUTION.md").read_text()+"\n\n"+(sources[1][1]/"ATTRIBUTION.md").read_text()+"\n\nFurther changes for this standalone study: prompt-only subsets, additional lexical filtering, and independent native-model rollouts. Original solutions are not used.\n")
    Path("data/CODEALPACA_LICENSE").write_bytes((sources[0][1]/"DATA_LICENSE").read_bytes())
    print(json.dumps({"selected": len(selected), "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
