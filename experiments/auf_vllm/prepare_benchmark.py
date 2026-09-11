"""Create an isolated benchmark view without modifying archived evidence."""
import argparse
import json
import os
from pathlib import Path
from .pilot_data import rows, write, sha


def main(case, original):
    assert (case/"export/export_check.json").exists()
    for size in [4,8]:
        for kind in ["target","draft"]:
            link = case/f"models/{size}b/{kind}"
            link.parent.mkdir(parents=True,exist_ok=True)
            expected = original/f"models/{size}b/{kind}"
            if link.exists():
                assert link.resolve() == expected.resolve()
            else:
                link.symlink_to(expected.resolve(),target_is_directory=True)
    manifest = original/"common/dataset/dev.jsonl"
    development = rows(manifest)
    evaluation, warmup = development[:128], development[-4:]
    assert len({x["group_id"] for x in evaluation}) == 128
    assert not ({x["group_id"] for x in evaluation} & {x["group_id"] for x in warmup})
    for name, data in [("eval",evaluation),("warmup",warmup)]:
        dest = case/f"evaluation/{name}.jsonl"
        dest.parent.mkdir(parents=True,exist_ok=True)
        content = "".join(json.dumps(x)+"\n" for x in data)
        if dest.exists():
            assert dest.read_text() == content
        else:
            dest.write_text(content)
    write(case/"evaluation/provenance.json", {"source":str(manifest),"sha256":sha(manifest),
          "split":"development only", "count":128,"warmup_count":4,"target_adapters":None})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",type=Path,required=True)
    args=parser.parse_args()
    main(args.case,Path(os.environ["TRANSFER_WORK"]))
