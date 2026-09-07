"""Bounded source-layer/context loss intervention with folded deployment export."""

import gc
import hashlib
import io
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from relayspec.layer_context import (
    LayerContextMapper,
    frozen_norm,
    layer_context_loss,
    relative_error,
)
from relayspec.relay import TargetFeatureRelay, extract_hidden_taps


def fit(spec, root, storage, config, draft):
    start = time.perf_counter()
    device = torch.device("cuda:0")
    settings = spec["layer_context_fit"]
    cache = Path(spec["feature_cache"])
    index_path = cache / "cache-index.json"
    index = json.loads(index_path.read_text())
    assert index["status"] == "pass"
    assert (
        hashlib.sha256(index_path.read_bytes()).hexdigest() == settings["cache_sha256"]
    )
    for key in ["target", "proposer", "source_trunk"]:
        assert index["metadata"]["config"][key] == config[key], key
    taps = list(index["metadata"]["target_layer_ids"])
    assert taps == [1, 9, 17, 25, 33] and list(draft.target_layer_ids) == taps
    fusion = draft.fc.weight.detach().to(device)
    gamma = draft.hidden_norm.weight.detach().to(device)
    assert fusion.shape == (2560, 12800) and draft.hidden_norm.variance_epsilon == 1e-6
    source_spec = config["source_trunk"]["model"]
    source = (
        AutoModelForCausalLM.from_pretrained(
            source_spec["id"],
            revision=source_spec["revision"],
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    data = {}
    selections = []
    check_errors = []
    rng = torch.Generator().manual_seed(settings["seed"])
    for split, count in [
        ("train", settings["records"]),
        ("validation", settings["validation_records"]),
    ]:
        entries = [e for e in index["entries"] if e["split"] == split][:count]
        assert len(entries) == count
        xs = []
        ys = []
        for e in entries:
            raw = (cache / e["file"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == e["sha256"]
            payload = torch.load(io.BytesIO(raw), weights_only=True)
            ids = payload["input_ids"].to(device)
            assert ids.shape[1] >= 32
            positions = torch.randperm(ids.shape[1], generator=rng)[:32].sort().values
            with torch.no_grad():
                output = source(
                    ids, use_cache=False, output_hidden_states=True, logits_to_keep=1
                )
                y = extract_hidden_taps(output.hidden_states, taps)[
                    0, positions.to(device)
                ].reshape(32, 5, 2560)
                if len(check_errors) < 8:
                    full = frozen_norm(
                        torch.nn.functional.linear(y.flatten(1), fusion), gamma
                    )
                    old = payload["y"][0, positions].to(device)
                    check_errors.append(float(relative_error(full, old)))
                del output
            xs.append(payload["x"][0, positions].reshape(32, 5, 4096))
            ys.append(y.cpu())
            selections.append(
                dict(
                    split=split,
                    index=e["index"],
                    record_sha256=e["record_sha256"],
                    cache_sha256=e["sha256"],
                    positions=positions.tolist(),
                    input_ids_sha256=hashlib.sha256(
                        payload["input_ids"].numpy().tobytes()
                    ).hexdigest(),
                )
            )
        data[split] = (torch.cat(xs).to(device), torch.cat(ys).to(device))
    assert max(check_errors) < 1e-3, check_errors
    del source, draft
    gc.collect()
    torch.cuda.empty_cache()
    extraction_seconds = time.perf_counter() - start
    torch.manual_seed(settings["seed"])
    mode = settings["mode"]
    if mode == "dense_context":
        model = TargetFeatureRelay(
            target_hidden_size=4096,
            num_taps=5,
            draft_hidden_size=2560,
            eps=1e-6,
            normalize_input=True,
        ).to(device)
    else:
        model = LayerContextMapper(5, 4096, 2560).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=0.0
    )

    def losses(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "dense_context":
                prediction = frozen_norm(
                    model(x.flatten(1).unsqueeze(0)).squeeze(0), gamma
                )
                with torch.no_grad():
                    target = frozen_norm(
                        torch.nn.functional.linear(y.flatten(1), fusion), gamma
                    )
                return torch.zeros((), device=device), relative_error(
                    prediction, target
                )
            return layer_context_loss(model(x), y, fusion, gamma)

    def validate():
        sums = [0.0, 0.0]
        count = 0
        with torch.no_grad():
            x, y = data["validation"]
            for j in range(0, len(x), 2048):
                a, b = losses(x[j : j + 2048], y[j : j + 2048])
                n = len(x[j : j + 2048])
                sums[0] += float(a) * n
                sums[1] += float(b) * n
                count += n
        return dict(layer=sums[0] / count, context=sums[1] / count)

    history = [dict(step=0, validation=validate())]
    x, y = data["train"]
    fitting_start = time.perf_counter()
    for step in range(1, settings["steps"] + 1):
        positions = (torch.arange(2048, device=device) + (step - 1) * 2048) % len(x)
        optimizer.zero_grad(set_to_none=True)
        a, b = losses(x[positions], y[positions])
        loss = a + b if mode == "layer_context" else a if mode == "layer_only" else b
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step in [1, 32, 128, settings["steps"]]:
            row = dict(
                step=step,
                layer=float(a.detach()),
                context=float(b.detach()),
                validation=validate(),
            )
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    fitting_seconds = time.perf_counter() - fitting_start
    with torch.no_grad():
        if mode == "dense_context":
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            fold_error = None
        else:
            weight = model.fold(fusion)
            test = x[:32].float()
            expected = torch.nn.functional.linear(
                model(test).flatten(1), fusion.float()
            )
            actual = torch.nn.functional.linear(test.flatten(1), weight)
            fold_error = float(relative_error(actual, expected))
            assert fold_error < 1e-8
            state = {"projection.weight": weight.cpu()}
    cp = dict(
        relay=state,
        target_layer_ids=taps,
        steps=settings["steps"],
        seed=settings["seed"],
        proposer_family="dflash",
        relay_architecture="normalized_linear"
        if mode == "dense_context"
        else "scale_preserving_linear",
        feature_objective=mode,
    )
    path = storage / "relay-layer-context.pt"
    torch.save(cp, path)
    torch.save(model.state_dict(), storage / "unfolded-mapper.pt")
    result = dict(
        status="pass",
        settings=settings,
        history=history,
        teacher_context_reproduction_errors=check_errors,
        fold_fp32_relative_error=fold_error,
        extraction_seconds=extraction_seconds,
        fitting_seconds=fitting_seconds,
        checkpoint=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parameter_count=sum(p.numel() for p in model.parameters()),
        source_model=source_spec,
        scope="512 records from existing192token prefixes;32 fixed positions per record;128 updates of2048 positions. Folded FP32 projection algebra verified; BF16 two-stage versus folded rounding is not bitwise identical. No full-length rollout reproduction claim.",
    )
    (root / "layer-context-fit.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "layer-context-selection.json").write_text(
        json.dumps(selections, indent=2) + "\n"
    )
    gc.collect()
    torch.cuda.empty_cache()
    return str(path), result
