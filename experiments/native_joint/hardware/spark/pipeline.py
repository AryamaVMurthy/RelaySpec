"""Bounded Spark deployment: verify assets, gate, measure, then profile kernels."""
import argparse
import datetime
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGET_REV = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_REV = "9b41424b7109f9c5413454f481b09a82b85333f4"
WEIGHTS = [
    "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--download-pid", type=int, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    out = base / "evaluation-01"
    out.mkdir(exist_ok=False)
    checkpoint = base / "selected-29162-lane0.pt"
    env = dict(os.environ, DFLASH_SOURCE=str(base / "vendor/dflash"),
               HF_HUB_CACHE=str(base / "hf-direct/hub"), CUDA_VISIBLE_DEVICES="0",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    os.environ.update(env)
    nsys = "/opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys"
    ncu = "/opt/nvidia/nsight-compute/2025.3.1/ncu"
    def status(stage, **values):
        row = dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
        temporary = out / "status.tmp"
        temporary.write_text(json.dumps(row, indent=2))
        temporary.replace(out / "status.json")
        print(json.dumps(row), flush=True)
    def idle():
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
        if active:
            raise RuntimeError("GPU has an existing compute process; refusing overlapping measurements: " + active)
    def call(stage, command, timeout):
        status(stage, command=command, timeout_seconds=timeout)
        (out / (stage + "-command.json")).write_text(json.dumps(command, indent=2))
        with (out / (stage + ".log")).open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if code:
            raise RuntimeError(f"{stage} failed with exit {code}; partial output preserved")
    def bench(config, output, *extra):
        return [sys.executable, str(HERE / "benchmark.py"), "--config", str(HERE / (config + ".json")),
                "--checkpoint", str(checkpoint), "--output", str(out / output), *extra]
    try:
        status("waiting_for_asset_transfer", download_pid=args.download_pid)
        deadline = time.monotonic() + 10800
        # The already-started downloader holds HF file locks. Let it finish,
        # then verify every required weight before any GPU experiment begins.
        while Path(f"/proc/{args.download_pid}/cmdline").exists():
            if b"snapshot_download" not in Path(f"/proc/{args.download_pid}/cmdline").read_bytes():
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Model transfer exceeded three hours")
            time.sleep(15)
        from huggingface_hub import snapshot_download
        for model, revision in [("Qwen/Qwen3-8B", TARGET_REV), ("z-lab/Qwen3-8B-DFlash-b16", DRAFT_REV)]:
            # Local-only lookup succeeds only once the original downloader has
            # published snapshots. Missing assets are an explicit failure.
            snapshot_download(model, revision=revision, cache_dir=env["HF_HUB_CACHE"], local_files_only=True)
        while not checkpoint.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Checkpoint transfer exceeded three hours")
            time.sleep(15)
        status("verifying_asset_sha256")
        cache = Path(env["HF_HUB_CACHE"])
        target = cache / "models--Qwen--Qwen3-8B/snapshots" / TARGET_REV
        draft = cache / "models--z-lab--Qwen3-8B-DFlash-b16/snapshots" / DRAFT_REV
        entries = [(target / f"model-{i:05}-of-00005.safetensors", h) for i, h in enumerate(WEIGHTS, 1)]
        entries += [(draft / "model.safetensors", "c702878094f38ad5843e6ac40b327720b3147ffbf669a5d9ca1974865ca6c080"),
                    (checkpoint, json.loads((HERE / "main.json").read_text())["checkpoint_sha256"])]
        verified = []
        for path, expected in entries:
            with path.open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            assert actual == expected, str(path)
            verified.append(dict(path=str(path), sha256=actual, bytes=path.stat().st_size))
        (out / "verified-assets.json").write_text(json.dumps(verified, indent=2))
        idle()
        call("smoke", bench("smoke", "smoke"), 1800)
        call("smoke-analysis", [sys.executable, str(HERE / "analyze.py"), str(out / "smoke")], 120)
        idle()
        call("timeline", [nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
             "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-o", str(out / "timeline"),
             *bench("profile", "timeline-run", "--profile")], 1800)
        call("timeline-stats", [nsys, "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum",
             "--format", "csv", "--output", str(out / "timeline-summary"), str(out / "timeline.nsys-rep")], 300)
        idle()
        # A generous hard limit prevents an accidental unbounded inference job.
        call("main", bench("main", "main"), 14400)
        call("main-analysis", [sys.executable, str(HERE / "analyze.py"), str(out / "main")], 300)
        # Bounded fixed sampling: initial projection kernels in each phase.
        # The timeline supports checking how representative these kernels are.
        for method in ["native", "reference"]:
            for phase in ["draft", "target_decode"]:
                idle()
                name = f"counters-{method}-{phase}"
                call(name, [ncu, "--profile-from-start", "off", "--nvtx", "--nvtx-include", phase + "/",
                     "--kernel-name-base", "demangled", "--kernel-name", "regex:.*(gemm|gemv|cutlass|Kernel2).*", "--set", "basic",
                     "--section", "MemoryWorkloadAnalysis", "--launch-count", "12", "-o", str(out / name),
                     *bench("counters", name + "-run", "--profile", "--profile-method", method)], 1200)
                call(name + "-csv", [ncu, "--import", str(out / (name + ".ncu-rep")), "--page", "raw", "--csv"], 120)
                csv_text = (out / (name + "-csv.log")).read_text()
                if '"Kernel Name"' not in csv_text or '"gpu__time_duration.sum"' not in csv_text:
                    raise RuntimeError(f"{name}: profiler did not export kernel duration counters")
        status("complete", throughput=str(out / "main/summary.json"),
               caveat="Hardware profiles are separate diagnostic passes, not throughput measurements")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
