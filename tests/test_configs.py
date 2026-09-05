import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_eagle_slurm_uses_pinned_transformers_overlay_explicitly() -> None:
    script = (ROOT / "slurm" / "benchmark_eagle3.sbatch").read_text(encoding="utf-8")
    assert ': "${DEEPSPEC_PYTHON_OVERLAY:?' in script
    assert (
        'export PYTHONPATH="$PROJECT_DIR/src:$DEEPSPEC_PYTHON_OVERLAY:'
        '$DEEPSPEC_SOURCE${PYTHONPATH:+:$PYTHONPATH}"'
    ) in script


def test_eagle_memory_relays_use_promoted_validation_checkpoints() -> None:
    for model_size in ("8b", "14b"):
        memory = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "protocol_active"
                / f"eagle3_qwen3_{model_size}_memory_relay_4gpu.yaml"
            ).read_text()
        )
        validated = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "protocol_active"
                / f"eagle3_qwen3_{model_size}_math500_source_full_4gpu.yaml"
            ).read_text()
        )
        assert (
            memory["relay_probe"]["checkpoint_path"]
            == validated["relay_probe"]["checkpoint_path"]
        )


def test_configs_use_four_gpus_and_non_thinking_qwen3() -> None:
    for name in ("train_relay_qwen3_4gpu.yaml", "benchmark_relay_qwen3_4gpu.yaml"):
        payload = yaml.safe_load((ROOT / "configs" / name).read_text())
        assert payload["resources"]["gpu_count"] == 4
        assert payload["generation"]["enable_thinking"] is False
        assert payload["target"]["id"] == "Qwen/Qwen3-14B"
        assert payload["proposer"]["id"] == "z-lab/Qwen3-4B-DFlash-b16"


def test_benchmark_is_paired_and_repeated() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "benchmark_relay_qwen3_4gpu.yaml").read_text()
    )
    assert payload["benchmark"]["benchmarks"] == ["math500"]
    assert payload["benchmark"]["max_prompts"] == 4
    assert payload["benchmark"]["methods"] == [
        "matched_full_target_dflash",
        "target_feature_relay",
    ]
    assert payload["relay_probe"]["repetitions"] == 3


def test_paper_benchmark_configs_compare_native_and_use_long_outputs() -> None:
    smoke = yaml.safe_load(
        (
            ROOT / "configs" / "benchmark_relay_qwen3_14b_math500_smoke_4gpu.yaml"
        ).read_text()
    )
    full = yaml.safe_load(
        (
            ROOT / "configs" / "benchmark_relay_qwen3_14b_math500_full_4gpu.yaml"
        ).read_text()
    )
    expected_methods = ["native_ar", "naive_source_reuse", "relay_p"]
    assert smoke["benchmark"]["methods"] == expected_methods
    assert full["benchmark"]["methods"] == expected_methods
    assert smoke["benchmark"]["max_prompts"] == 8
    assert smoke["generation"]["max_new_tokens"] == 256
    assert full["benchmark"]["max_prompts"] == 500
    assert full["generation"]["max_new_tokens"] == 2048
    assert smoke["generation"]["enable_thinking"] is False
    assert full["generation"]["enable_thinking"] is False
    assert smoke["resources"]["gpu_count"] == 4
    assert full["resources"]["gpu_count"] == 4


def test_qwen3_8b_controlled_configs_include_native_dflash_ceiling() -> None:
    training = yaml.safe_load(
        (ROOT / "configs" / "train_relay_qwen3_8b_4gpu.yaml").read_text()
    )
    smoke = yaml.safe_load(
        (
            ROOT / "configs" / "benchmark_relay_qwen3_8b_math500_smoke_4gpu.yaml"
        ).read_text()
    )
    full = yaml.safe_load(
        (
            ROOT / "configs" / "benchmark_relay_qwen3_8b_math500_full_4gpu.yaml"
        ).read_text()
    )
    expected = [
        "native_ar",
        "native_target_dflash",
        "naive_source_reuse",
        "relay_f",
    ]
    assert training["resources"]["gpu_count"] == 4
    assert training["target"]["id"] == "Qwen/Qwen3-8B"
    assert training["generation"]["enable_thinking"] is False
    assert smoke["benchmark"]["methods"] == expected
    assert full["benchmark"]["methods"] == expected
    assert smoke["native_target_proposer"]["id"] == "z-lab/Qwen3-8B-DFlash-b16"
    assert full["generation"]["max_new_tokens"] == 2048
    assert full["benchmark"]["max_prompts"] == 500


def test_core_suite_configs_cover_fixed_gsm_and_code_manifests() -> None:
    for model_size in ("8b", "14b"):
        payload = yaml.safe_load(
            (
                ROOT
                / "configs"
                / f"benchmark_relay_qwen3_{model_size}_core_suite_4gpu.yaml"
            ).read_text()
        )
        assert payload["benchmark"]["benchmarks"] == [
            "gsm8k",
            "humaneval",
            "mbpp",
        ]
        assert payload["benchmark"]["max_prompts"] == 128 + 164 + 200
        assert payload["generation"]["max_new_tokens"] == 2048
        assert payload["generation"]["enable_thinking"] is False
        assert payload["resources"]["gpu_count"] == 4


def test_mtbench_configs_run_all_eighty_two_turn_records() -> None:
    for model_size in ("8b", "14b"):
        payload = yaml.safe_load(
            (
                ROOT
                / "configs"
                / f"benchmark_relay_qwen3_{model_size}_mtbench_4gpu.yaml"
            ).read_text()
        )
        assert payload["benchmark"]["benchmarks"] == ["mtbench"]
        assert payload["benchmark"]["max_prompts"] == 80
        assert payload["generation"]["max_new_tokens"] == 2048
        assert payload["generation"]["enable_thinking"] is False


def test_corrected_mbpp_configs_use_pinned_evalplus_manifest() -> None:
    manifest_path = ROOT / "configs" / "eval_manifest_full_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [r for r in manifest["records"] if r["benchmark"] == "mbpp"]
    assert len(records) == 200
    assert len({r["problem_id"] for r in records}) == 200
    assert all(r["problem_id"].startswith("Mbpp/") for r in records)
    assert all("Your code must satisfy all base tests:" in r["prompt"] for r in records)
    assert manifest["dataset_revisions"]["evalplus_mbpp"] == (
        "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
    )
    for model_size in ("8b", "14b"):
        payload = yaml.safe_load(
            (
                ROOT
                / "configs"
                / f"benchmark_relay_qwen3_{model_size}_mbpp_corrected_4gpu.yaml"
            ).read_text()
        )
        assert payload["benchmark"]["manifest_path"].endswith(
            "eval_manifest_full_v3.json"
        )
        assert payload["benchmark"]["benchmarks"] == ["mbpp"]
        assert payload["benchmark"]["max_prompts"] == 200
        assert payload["resources"]["gpu_count"] == 4

    full = json.loads(
        (ROOT / "configs" / "eval_manifest_full_v4.json").read_text(encoding="utf-8")
    )
    remainder = json.loads(
        (ROOT / "configs" / "eval_manifest_mbpp_remainder.json").read_text(
            encoding="utf-8"
        )
    )
    first_ids = {r["problem_id"] for r in records}
    full_ids = {r["problem_id"] for r in full["records"] if r["benchmark"] == "mbpp"}
    remainder_ids = {
        r["problem_id"] for r in remainder["records"] if r["benchmark"] == "mbpp"
    }
    assert len(full_ids) == 378
    assert len(remainder_ids) == 178
    assert not first_ids & remainder_ids
    assert first_ids | remainder_ids == full_ids


def test_proposal_refinement_is_initialized_from_feature_relay() -> None:
    training = yaml.safe_load(
        (ROOT / "configs" / "train_relay_qwen3_8b_proposal_4gpu.yaml").read_text()
    )
    assert training["relay_training"]["proposal_kl_weight"] > 0
    assert training["relay_training"]["proposal_block_size"] == 16
    assert (
        "qwen3-8b-dflash4b-feature"
        in training["relay_training"]["initial_checkpoint_path"]
    )
    for size in ("smoke", "full"):
        payload = yaml.safe_load(
            (
                ROOT
                / "configs"
                / f"benchmark_relay_qwen3_8b_proposal_math500_{size}_4gpu.yaml"
            ).read_text()
        )
        assert "relay_p" in payload["benchmark"]["methods"]
        assert "qwen3-8b-dflash4b-proposal" in payload["relay_probe"]["checkpoint_path"]


def test_conservative_proposal_refinement_preserves_feature_solution() -> None:
    payload = yaml.safe_load(
        (
            ROOT / "configs" / "train_relay_qwen3_8b_proposal_conservative_4gpu.yaml"
        ).read_text()
    )
    training = payload["relay_training"]
    assert payload["resources"]["gpu_count"] == 4
    assert training["initial_checkpoint_path"].endswith(
        "qwen3-8b-dflash4b-feature/relay.pt"
    )
    assert training["proposal_kl_weight"] == 0.1
    assert training["learning_rate"] == 0.000001
    assert training["steps"] == 1024


def test_conservative_proposal_full_eval_uses_promoted_checkpoint() -> None:
    payload = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "benchmark_relay_qwen3_8b_proposal_conservative_math500_full_4gpu.yaml"
        ).read_text()
    )
    assert payload["resources"]["gpu_count"] == 4
    assert payload["generation"]["enable_thinking"] is False
    assert payload["generation"]["max_new_tokens"] == 2048
    assert payload["benchmark"]["benchmarks"] == ["math500"]
    assert payload["benchmark"]["methods"] == [
        "native_ar",
        "native_target_dflash",
        "naive_source_reuse",
        "relay_p",
    ]
    assert payload["benchmark"]["max_prompts"] == 500
    assert payload["relay_probe"]["checkpoint_path"].endswith(
        "qwen3-8b-dflash4b-proposal-conservative/relay.pt"
    )


def test_memory_ablation_configs_isolate_source_trunk_residency() -> None:
    loaded = yaml.safe_load(
        (
            ROOT / "configs" / "benchmark_relay_qwen3_8b_memory_loaded_4gpu.yaml"
        ).read_text()
    )
    unloaded = yaml.safe_load(
        (
            ROOT / "configs" / "benchmark_relay_qwen3_8b_memory_unloaded_4gpu.yaml"
        ).read_text()
    )
    for payload in (loaded, unloaded):
        assert payload["resources"]["gpu_count"] == 4
        assert payload["benchmark"]["methods"] == ["native_ar", "relay_f"]
        assert payload["benchmark"]["max_prompts"] == 32
        assert payload["generation"]["enable_thinking"] is False
    assert loaded["benchmark"]["unload_source_trunk"] is False
    assert unloaded["benchmark"]["unload_source_trunk"] is True


def test_block_size_ablation_changes_only_the_speculative_block() -> None:
    payloads = []
    for block_size in (8, 16, 32):
        payload = yaml.safe_load(
            (
                ROOT
                / "configs"
                / f"benchmark_relay_qwen3_8b_block{block_size}_4gpu.yaml"
            ).read_text()
        )
        assert payload["resources"]["gpu_count"] == 4
        assert payload["generation"]["enable_thinking"] is False
        assert payload["generation"]["max_new_tokens"] == 512
        assert payload["benchmark"]["max_prompts"] == 64
        assert payload["benchmark"]["block_size"] == block_size
        payloads.append(payload)
    comparable = [
        {
            **payload,
            "run_name": "normalized",
            "benchmark": {**payload["benchmark"], "block_size": 0},
        }
        for payload in payloads
    ]
    assert comparable[0] == comparable[1] == comparable[2]
