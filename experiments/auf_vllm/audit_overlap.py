"""Exact/template overlap audit against the paper's archived benchmarks."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from .pilot_data import write,sha


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC",text).lower().split())


def group(text):
    text=re.sub(r"\d+(?:\.\d+)?","#",normalize(text))
    text=re.sub(r"[^\w#]+"," ",text)
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def main(args):
    training=[]
    with args.train.open() as stream:
        for index,line in enumerate(stream):
            row=json.loads(line)
            assert group(row["problem"]) == row["group_id"]
            training.append((row["group_id"],normalize(row["problem"])))
    benchmarks=json.loads(args.benchmarks.read_text())["records"]
    output=[]
    for count in [4096,16384]:
        assert len(training)>=count
        groups={x[0] for x in training[:count]}
        exact={x[1] for x in training[:count]}
        hits=[]
        for row in benchmarks:
            prompts=[row["prompt"]] if "prompt" in row else row["turns"]
            assert all(isinstance(prompt,str) for prompt in prompts)
            exact_match=any(normalize(prompt) in exact for prompt in prompts)
            template_match=any(group(prompt) in groups for prompt in prompts)
            if exact_match or template_match:
                hits.append({"benchmark":row["benchmark"],"problem_id":row["problem_id"],
                             "exact":exact_match,"template":template_match})
        output.append({"training_records":count,"matches":hits})
    write(args.out,{"training_sha256":sha(args.train),"benchmarks_sha256":sha(args.benchmarks),
                   "scope":"Exact and conservative number/punctuation template overlap; not semantic decontamination",
                   "results":output})
    print(json.dumps([{ "training_records":x["training_records"],"matches":len(x["matches"])} for x in output]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",type=Path,required=True)
    parser.add_argument("--benchmarks",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
