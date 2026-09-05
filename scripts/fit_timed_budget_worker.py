"""Dispatch one feature fit and three adaptation fits within four allocated GPUs."""

import argparse
import os
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--trials", required=True)
args = parser.parse_args()
if int(os.environ["LOCAL_RANK"]) == 0:
    os.environ["RELAYSPEC_OUTPUT"] = os.environ["RELAYSPEC_FIT_OUTPUT"]
    sys.argv = [
        "scripts/fit_cached_mappers.py",
        "--trials",
        args.trials,
        "--single-trial",
    ]
else:
    sys.argv = ["scripts/fit_drafter_adaptation_pilot.py", "--config", args.config]
runpy.run_path(sys.argv[0], run_name="__main__")
