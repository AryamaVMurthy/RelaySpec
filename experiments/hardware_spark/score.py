"""Local answer/code scoring through the existing isolated, resource-limited grader."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCORER = HERE.parent / "native_joint/analysis/quality.py"
sys.path.insert(0, str(SCORER.parent))
from quality import score, QWEN_EVAL, PRIVATE_DEPS


def main():
    p = argparse.ArgumentParser(); p.add_argument("run",type=Path)
    args=p.parse_args(); out=args.run
    done=json.loads((out/"complete.json").read_text())
    assert done["status"]=="pass" and done["stage"]=="main"
    records={r["problem_id"]:r for r in json.loads((out/"records.json").read_text())}
    rows=[r for r in map(json.loads,(out/"evaluation.jsonl").read_text().splitlines()) if r["repeat"]==0]
    assert len(rows)==len({(r["method"],r["problem_id"]) for r in rows})==len(records)*4
    unique={(r["problem_id"],r["completion"]):records[r["problem_id"]] for r in rows}
    def grade(item):
        (pid,completion),record=item
        return (pid,completion),score(record,completion)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=dict(pool.map(grade,unique.items()))
    scored=[dict(method=r["method"],problem_id=r["problem_id"],benchmark=r["benchmark"],
                 output_sha256=hashlib.sha256(json.dumps(r["tokens"]).encode()).hexdigest(),
                 **results[r["problem_id"],r["completion"]]) for r in rows]
    (out/"quality.jsonl").write_text("\n".join(map(json.dumps,scored))+"\n")
    summary={}
    for m in sorted({r["method"] for r in rows}):
        summary[m]={}
        for b in sorted({r["benchmark"] for r in rows}):
            group=[r for r in scored if r["method"]==m and r["benchmark"]==b]
            summary[m][b]={"correct":sum(r["correct"] is True for r in group),
                           "incorrect":sum(r["correct"] is False for r in group),
                           "unscored":sum(r["correct"] is None for r in group),"requests":len(group)}
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [SCORER,QWEN_EVAL/"grader.py",QWEN_EVAL/"parser.py"]}
    provenance={"scores":summary,"unique_completions_scored":len(unique),"code_hashes":hashes,
                "grader_dependencies":sorted(f.name for f in PRIVATE_DEPS.glob("*.dist-info")),
                "scope":"First repeat only after exact repeat audit. Identical problem/completion pairs share a deterministic score. No MT-Bench judge score; resource/infrastructure failures remain unscored."}
    (out/"quality-summary.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(summary,indent=2))
    failures=[r for r in scored if r["benchmark"]!="mtbench" and r["correct"] is None]
    if failures: raise RuntimeError(f"{len(failures)} non-dialogue outputs remain unscored; inspect quality.jsonl")


if __name__=="__main__": main()
