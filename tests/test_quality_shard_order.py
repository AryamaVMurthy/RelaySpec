import random
from pathlib import Path


def test_sharding_preserves_original_selected_requests_under_runtime_shuffle(
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from build_eagle_quality_campaign import manifest_for_order

    original = [{"problem_id": str(i)} for i in range(500)]
    random.Random(1729).shuffle(original)
    selected = original[:128]
    output = []
    for records in [selected[:64], selected[64:]]:
        manifest = manifest_for_order(records, 1729)
        random.Random(1729).shuffle(manifest)
        assert manifest == records
        output.extend(manifest)
    assert output == selected
    assert len({r["problem_id"] for r in output}) == 128
    pilot = manifest_for_order(selected[:8], 1729)
    random.Random(1729).shuffle(pilot)
    assert pilot == selected[:8]
