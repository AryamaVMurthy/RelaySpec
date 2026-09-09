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
    p.add_argument("--first-pass",action="store_true",help="Grade all 32 completed first-pass requests while timing repeats are still running")
    args=p.parse_args(); out=args.run
    complete=out/"complete.json"
    if complete.exists():
        done=json.loads(complete.read_text())
        assert done["status"]=="pass" and done["stage"]=="main"
    else:
        assert args.first_pass,"Timing gate is incomplete; explicit first-pass mode is required"
    provenance=json.loads((out/"provenance.json").read_text())
    assert provenance["stage"]=="main" and provenance["requests"]==32 and provenance["cap"]==2048
    records={r["problem_id"]:r for r in json.loads((out/"records.json").read_text())}
    lines=(out/"evaluation.jsonl").read_text().splitlines(keepends=True)
    if args.first_pass and lines and not lines[-1].endswith("\n"):
        lines=lines[:-1]  # A live second-pass write is outside the grading set.
    rows=[r for r in map(json.loads,lines) if r["repeat"]==0]
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
    scoring_inputs={"records":records,"outputs":[{k:r[k] for k in ["method","problem_id","tokens","completion"]} for r in sorted(rows,key=lambda r:(r["method"],r["problem_id"]))]}
    provenance={"scores":summary,"unique_completions_scored":len(unique),"code_hashes":hashes,
                "scoring_inputs_sha256":hashlib.sha256(json.dumps(scoring_inputs,sort_keys=True).encode()).hexdigest(),
                "timing_gate_at_scoring":"pass" if complete.exists() else "pending; paper builder must validate completed repeat gate and scoring input hash",
                "grader_dependencies":sorted(f.name for f in PRIVATE_DEPS.glob("*.dist-info")),
                "scope":"All 32 first-pass requests. Identical problem/completion pairs share a deterministic score. Final paper assets require the completed timing/repeat gate and identical scoring inputs. No MT-Bench judge score; resource/infrastructure failures remain unscored."}
    (out/"quality-summary.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(summary,indent=2))
    failures=[r for r in scored if r["benchmark"]!="mtbench" and r["correct"] is None]
    if failures: raise RuntimeError(f"{len(failures)} non-dialogue outputs remain unscored; inspect quality.jsonl")


if __name__=="__main__": main()
