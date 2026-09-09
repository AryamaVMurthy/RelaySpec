"""Bounded, sequential real-GPU paper evaluation after a verified smoke run."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    output = base / "paper-run-01"
    output.mkdir(exist_ok=False)
    env = dict(os.environ, DFLASH_SOURCE="/home/sarcs/native-spark-20260909/vendor/dflash",
        HF_HUB_CACHE="/home/sarcs/native-spark-20260909/hf-direct/hub", SOURCE_HF_CACHE="/home/sarcs/.cache/huggingface/hub",
        CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    nsys = "/opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys"
    ncu = "/opt/nvidia/nsight-compute/2025.3.1/ncu"
    def status(stage, **details):
        data = dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details)
        (output / "status.tmp").write_text(json.dumps(data,indent=2))
        (output / "status.tmp").replace(output / "status.json")
        print(json.dumps(data), flush=True)
    def invoke(stage, cmd, timeout):
        status(stage, command=cmd, timeout_seconds=timeout)
        (output / (stage+"-command.json")).write_text(json.dumps(cmd,indent=2))
        with (output / (stage+".log")).open("x") as log:
            process = subprocess.Popen(cmd,cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                raise
        if code:
            raise RuntimeError(f"{stage} failed with exit {code}; partial results preserved")
    def idle():
        active = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True).strip()
        if active: raise RuntimeError("GPU already has a compute process: "+active)
    def run(stage, directory, method="all"):
        return [sys.executable,str(HERE/"run.py"),"--stage",stage,"--method",method,
                "--checkpoint",str(base/"relay-primary.pt"),"--output",str(output/directory)]
    try:
        smoke=json.loads((base/"smoke-01/complete.json").read_text())
        assert smoke["status"]=="pass" and smoke["rows"]==16 and smoke["repeat_identity"]=="pass"
        invoke("smoke-analysis",[sys.executable,str(HERE/"analyze.py"),str(base/"smoke-01")],120)
        idle(); invoke("main",run("main","main"),14400)
        invoke("main-analysis",[sys.executable,str(HERE/"analyze.py"),str(output/"main")],300)
        for method in ["source","relay"]:
            idle(); name="memory-"+method
            invoke(name,run("memory",name,method),1800)
            invoke(name+"-analysis",[sys.executable,str(HERE/"analyze.py"),str(output/name)],120)
        idle()
        invoke("timeline",[nsys,"profile","--trace=cuda,nvtx","--sample=none","--cpuctxsw=none",
            "--capture-range=cudaProfilerApi","--capture-range-end=stop","-o",str(output/"timeline"),
            *run("profile","timeline-run")],1800)
        invoke("timeline-stats",[nsys,"stats","--report","cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum",
            "--format","csv","--output",str(output/"timeline-summary"),str(output/"timeline.nsys-rep")],300)
        for method, phase in [("source","source::verification_source_trunk"),("relay","relay::relay")]:
            idle(); name="counters-"+method
            invoke(name,[ncu,"--profile-from-start","off","--nvtx","--nvtx-include",phase+"/",
                "--kernel-name-base","demangled","--kernel-name","regex:.*(gemm|gemv|cutlass|Kernel2).*",
                "--set","basic","--section","MemoryWorkloadAnalysis","--launch-count","8","-o",str(output/name),
                *run("counters",name+"-run",method)],1800)
            invoke(name+"-csv",[ncu,"--import",str(output/(name+".ncu-rep")),"--page","raw","--csv"],120)
            csv=(output/(name+"-csv.log")).read_text()
            assert '"Kernel Name"' in csv and '"gpu__time_duration.sum"' in csv
        status("complete",scope="GPU measurements complete; local quality scoring and paper integration remain")
    except BaseException as error:
        status("failed",error=repr(error))
        raise


if __name__ == "__main__": main()
