"""Build a provenance-checked, domain-interleaved index of complete rollout caches."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path

from run_lane import TARGET, DRAFT, COMMIT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seen, domains, sources = set(), defaultdict(list), []
    sequence_lengths = set()
    for path in args.inputs:
        data = json.loads(path.read_text())
        if data["status"] != "pass" or data["target"] != TARGET or data["draft"] != DRAFT or data["source_commit"] != COMMIT:
            raise RuntimeError("Incomplete or incompatible rollout cache")
        gate = json.loads((path.parent/"sequence-causality-gate.json").read_text())
        if gate["status"] != "pass":
            raise RuntimeError("Missing prefix causality gate")
        sequence_lengths.add(data["config"]["sequence_length"])
        sources.append({"path": str(path), "sha256": sha(path), "excluded": data["excluded"]})
        for entry in data["entries"]:
            key = entry["question_sha"]
            if key in seen:
                raise RuntimeError("Repeated question within or across splits")
            seen.add(key)
            if len(entry["input_ids"]) != entry["tokens"] or entry["anchor_max"] != entry["tokens"]-16 or entry["anchor_min"] > entry["anchor_max"]:
                raise RuntimeError("Invalid sequence cache alignment")
            domains[entry["domain"]].append(entry)
    if len(sequence_lengths) != 1:
        raise RuntimeError("Sequence caps differ")
    entries = []
    for group in itertools.zip_longest(*(domains[k] for k in sorted(domains))):
        for entry in group:
            if entry is not None:
                entries.append({**entry, "index": len(entries)})
    counts = Counter(e["split"] for e in entries)
    if set(counts) != {"train", "validation"}:
        raise RuntimeError("Both data splits are required")
    index = {"status": "pass", "target": TARGET, "draft": DRAFT, "source_commit": COMMIT, "source_job": "native-rollouts", "sequence_length": sequence_lengths.pop(), "entries": entries, "sources": sources, "counts": dict(counts), "scope": "Fresh causal target features on frozen-native generated sequences. Domain-interleaved index. Raw tensor SHA256 is verified by the training runner before loading."}
    args.output.write_text(json.dumps(index, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "counts": dict(counts), "domains": {k: len(v) for k, v in domains.items()}}))


if __name__ == "__main__":
    main()
