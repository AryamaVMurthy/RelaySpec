"""One bounded, single-GPU research screen inside a four-GPU allocation."""

import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import yaml

from relayspec.ar_paper_evidence import summarize


def prepare(spec, root):
    storage = root / "weights"
    if os.environ.get("RELAYSPEC_CACHE_DIR"):
        storage = (
            Path(os.environ["RELAYSPEC_CACHE_DIR"])
            / "relayspec/autoresearch"
            / os.environ["SLURM_JOB_ID"]
            / root.name
        )
    storage.mkdir(parents=True, exist_ok=True)
    if spec["transform"] == "native_svd":
        from relayspec.dflash import import_official_dflash

        config = yaml.safe_load(Path(spec["config"]).read_text())
        klass, _ = import_official_dflash(
            os.environ["DFLASH_SOURCE"], config["proposer"]["source_commit"]
        )
        draft = klass.from_pretrained(
            config["proposer"]["id"],
            revision=config["proposer"]["revision"],
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        native = {
            "target_layer_ids": list(draft.target_layer_ids),
            "relay": {"projection.weight": draft.fc.weight.detach().float().cpu()},
            "relay_architecture": "scale_preserving_linear",
            "steps": 0,
            "origin": "Released native drafter fc; no fitting",
        }
        native_path = storage / "native-fc.pt"
        torch.save(native, native_path)
        spec = {**spec, "checkpoint": str(native_path)}
        del draft
    source = Path(spec["checkpoint"])
    base = torch.load(source, map_location="cpu", weights_only=True)
    weight = base["relay"]["projection.weight"]
    provenance = {
        "parent_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shape": list(weight.shape),
        "transform": spec["transform"],
    }
    variants = {"relay_base": str(source)}

    def save(name, checkpoint):
        path = storage / (name + ".pt")
        torch.save(checkpoint, path)
        variants[name] = str(path)

    if spec["transform"] in {"svd", "native_svd"}:
        torch.manual_seed(1729)
        w = weight.float().cuda()
        u, s, v = torch.svd_lowrank(
            w, q=min(max(spec["ranks"]) + 32, min(w.shape)), niter=4
        )
        provenance["randomized_svd"] = {
            "seed": 1729,
            "power_iterations": 4,
            "q": len(s),
        }
        provenance["relative_weight_errors"] = {}
        for rank in spec["ranks"]:
            cp = copy.deepcopy(base)
            cp["factorized_rank"] = rank
            cp["relay"] = {
                "projection.0.weight": v[:, :rank].T.contiguous().cpu(),
                "projection.1.weight": (u[:, :rank] * s[:rank]).cpu(),
            }
            provenance["relative_weight_errors"][str(rank)] = float(
                (w - (u[:, :rank] * s[:rank]) @ v[:, :rank].T).norm() / w.norm()
            )
            save(f"relay_svd{rank}", cp)
    elif spec["transform"] == "activation_svd":
        from relayspec.research_compression import activation_compression

        states, diagnostics = activation_compression(
            base, spec["feature_cache"], spec["ranks"]
        )
        provenance["activation_compression"] = diagnostics
        w = weight.float().cuda()
        torch.manual_seed(1729)
        u, singular, v = torch.svd_lowrank(
            w, q=min(max(spec["ranks"]) + 32, min(w.shape)), niter=4
        )
        for rank in spec["ranks"]:
            cp = copy.deepcopy(base)
            cp["factorized_rank"] = rank
            cp["relay"] = {
                "projection.0.weight": v[:, :rank].T.contiguous().cpu(),
                "projection.1.weight": (u[:, :rank] * singular[:rank]).cpu(),
            }
            save(f"relay_weight{rank}", cp)
        for rank, state in states.items():
            cp = copy.deepcopy(base)
            cp["factorized_rank"] = rank
            cp["relay"] = state
            save(f"relay_activation{rank}", cp)
    elif spec["transform"] == "drop_taps":
        taps = base["target_layer_ids"]
        width = weight.shape[1] // len(taps)
        provenance["target_layer_ids"] = list(taps)
        provenance["interpretation"] = (
            "Zero weight blocks; all taps still captured and input normalization unchanged. Causal feature-use ablation, not a memory-saving implementation."
        )
        for i, layer in enumerate(taps):
            cp = copy.deepcopy(base)
            cp["relay"]["projection.weight"][:, i * width : (i + 1) * width] = 0
            save(f"relay_drop{layer}", cp)
    elif spec["transform"] == "fit_reduced_taps":
        from relayspec.scaling_matrix import validate_cache_access_gate, validate_trial

        cache = Path(spec["feature_cache"])
        cache_hash = hashlib.sha256(
            (cache / "cache-index.json").read_bytes()
        ).hexdigest()
        pilot = json.loads(Path(spec["pilot_gate"]).read_text())
        if (
            pilot["status"] != "pass"
            or pilot["feature_cache_index_sha256"] != cache_hash
        ):
            raise ValueError("Reduced-tap fit requires a passed exact-cache pilot")
        trial = copy.deepcopy(spec["trial"])
        validate_trial(trial)
        access = (
            json.loads(Path(spec["access_gate"]).read_text())
            if spec.get("access_gate")
            else None
        )
        validate_cache_access_gate([trial], access, cache_hash)
        trials_path = root / "trials.json"
        trials_path.write_text(json.dumps({"trials": [trial]}, indent=2) + "\n")
        fit_root = storage / "fitting"
        env = {
            **os.environ,
            "RELAYSPEC_OUTPUT": str(fit_root),
            "RELAYSPEC_FEATURE_CACHE": str(cache),
            "RELAYSPEC_RESEARCH_SINGLE_GPU": "1",
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
        }
        subprocess.run(
            [
                os.environ["RELAYSPEC_PYTHON"],
                "scripts/fit_cached_mappers.py",
                "--single-trial",
                "--equivalence-pilot",
                "--trials",
                str(trials_path),
            ],
            env=env,
            check=True,
        )
        complete = json.loads(
            (fit_root / trial["name"] / "fit-complete.json").read_text()
        )
        if (
            complete["status"] != "pass"
            or complete["distinct_records_seen"] != trial["distinct_examples"]
        ):
            raise ValueError("Reduced-tap fitting did not complete")
        path = fit_root / trial["name"] / f"step-{trial['steps']:06d}.pt"
        cp = torch.load(path, weights_only=True, map_location="cpu")
        if cp["target_layer_ids"] != trial["target_layer_ids"]:
            raise ValueError("Exported mapper lost selected taps")
        import shutil

        diagnostics = root / "fitting" / trial["name"]
        diagnostics.mkdir(parents=True, exist_ok=True)
        for artifact in (fit_root / trial["name"]).iterdir():
            if artifact.suffix != ".pt":
                shutil.copy2(artifact, diagnostics / artifact.name)
        variants["relay_reduced"] = str(path)
        provenance["fit_trial"] = trial
        provenance["cache_sha256"] = cache_hash
    elif spec["transform"] == "existing":
        for name, candidate in spec["candidates"].items():
            if not Path(candidate).is_file():
                raise FileNotFoundError(candidate)
            variants[name] = candidate
        provenance["candidates"] = spec["candidates"]
    elif spec["transform"] == "joint_drop":
        taps = list(base["target_layer_ids"])
        width = weight.shape[1] // len(taps)
        for group in spec["drop_groups"]:
            if not set(group).issubset(taps):
                raise ValueError("Intervention layer absent from mapper")
            cp = copy.deepcopy(base)
            for layer in group:
                i = taps.index(layer)
                cp["relay"]["projection.weight"][:, i * width : (i + 1) * width] = 0
            save("relay_drop_" + "_".join(map(str, group)), cp)
        provenance["interpretation"] = (
            "Joint block masking at fixed input normalization and feature capture."
        )
    elif spec["transform"] == "quantization_noise":
        for bits in [2, 4, 8]:
            levels = 2 ** (bits - 1) - 1
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / levels
            cp = copy.deepcopy(base)
            cp["relay"]["projection.weight"] = (weight / scale).round().clamp(
                -levels, levels
            ) * scale
            save(f"relay_fakeq{bits}", cp)
        provenance["interpretation"] = (
            "Per-row symmetric quantize/dequantize weights, evaluated in BF16. This tests numerical robustness, not integer-kernel speed or memory savings."
        )
    elif spec["transform"] == "data_interpolation":
        small_path = Path(spec["small_checkpoint"])
        small = torch.load(small_path, map_location="cpu", weights_only=True)
        assert small["target_layer_ids"] == base["target_layer_ids"]
        provenance["small_sha256"] = hashlib.sha256(small_path.read_bytes()).hexdigest()
        variants["relay_small"] = str(small_path)
        for alpha in [0.25, 0.5, 0.75]:
            cp = copy.deepcopy(base)
            cp["relay"]["projection.weight"] = weight * alpha + small["relay"][
                "projection.weight"
            ] * (1 - alpha)
            save(f"relay_blend{int(alpha * 100)}", cp)
    else:
        raise ValueError("Unknown research transform")
    (root / "transform.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return variants


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    for checkpoint, expected in spec.get("checkpoint_sha256", {}).items():
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen checkpoint hash mismatch: {checkpoint}")
    variants = prepare(spec, a.output)
    gc.collect()
    torch.cuda.empty_cache()
    config = yaml.safe_load(Path(spec["config"]).read_text())
    family = config["proposer"]["family"]
    config["resources"]["gpu_count"] = 1
    config["benchmark"].update(
        max_prompts=spec.get("requests", 8),
        research_single_gpu=True,
        methods=[
            *(["native_ar"] if spec.get("include_ar", True) else []),
            *(
                [
                    "native_target_dflash"
                    if family == "dflash"
                    else "native_target_eagle3"
                ]
                if spec.get("include_native", True)
                else []
            ),
            *variants,
        ],
    )
    config["benchmark"].pop("isolate_methods", None)
    if family == "dflash":
        config["benchmark"]["unload_source_trunk"] = True
    config["generation"]["max_new_tokens"] = spec.get("max_new_tokens", 128)
    config["relay_probe"]["variants"] = variants
    cfg = a.output / "config.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    py = os.environ["RELAYSPEC_PYTHON"]
    env = {**os.environ, "RELAYSPEC_OUTPUT": str(a.output)}
    subprocess.run(
        [
            py,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "scripts/benchmark_relay.py"
            if family == "dflash"
            else "scripts/benchmark_eagle3.py",
            "--config",
            str(cfg),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            py,
            "scripts/check_run_complete.py",
            "--config",
            str(cfg),
            "--output",
            str(a.output),
        ],
        check=True,
    )
    rows = [
        json.loads(line)
        for line in (a.output / "benchmark-rank0.jsonl").read_text().splitlines()
    ]
    report = {
        "status": "pass",
        "hypothesis": spec["hypothesis"],
        "summary": summarize(rows, reference="relay_base"),
        "scope": spec.get(
            "scope",
            "Exposed development questions at the recorded output cap; exploratory screen, no full-answer quality or confirmatory significance claim.",
        ),
    }
    if spec.get("duplicate_control"):
        expected = {r["problem_id"]: r for r in rows if r["method"] == "relay_base"}
        differences = []
        for row in rows:
            if row["method"] == spec["duplicate_control"]:
                for field in [
                    "output_hash",
                    "acceptance_lengths",
                    "target_calls",
                    "draft_calls",
                ]:
                    if row[field] != expected[row["problem_id"]][field]:
                        differences.append(
                            {"problem_id": row["problem_id"], "field": field}
                        )
        report["duplicate_control"] = {
            "status": "pass" if not differences else "failed",
            "differences": differences,
        }
    (a.output / "research-result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
