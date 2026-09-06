import copy
import json
from pathlib import Path

import pytest


def test_paper_export_refuses_real_pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from build_eagle_quality_paper_assets import quality_table

    pilot = json.loads(
        Path(
            "reports/mapper-scaling-20260905/eagle3-small-quality-pilot-results.json"
        ).read_text()
    )
    with pytest.raises(ValueError, match="all 128"):
        quality_table(pilot)


def test_table_preserves_methods_and_rejects_omission(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from build_eagle_quality_paper_assets import quality_table

    # Synthetic formatting fixture only. It is never written to a result registry.
    protocol = json.loads(
        Path(
            "configs/submission/scaling/eagle3-small-quality-v1/protocol.json"
        ).read_text()
    )
    methods = protocol["methods"]
    fixture = {
        "status": "complete",
        "stage": "full",
        "requests": 128,
        "rows": 1408,
        "methods": {
            m: {"correct_count": 1, "cap_length_outputs": 127} for m in methods
        },
        "comparisons": {
            protocol["dense_reference"]: {
                "paired_throughput": {
                    "methods": {
                        m: {
                            "tokens_per_second": 10,
                            "throughput_ratio": 0.5,
                            "throughput_ci95": [0.4, 0.6],
                        }
                        for m in methods
                    }
                }
            }
        },
    }
    table = quality_table(fixture)
    assert table.count("1/128") == 11
    assert table.count("50.00 [40.00, 60.00]") == 11
    changed = copy.deepcopy(fixture)
    changed["methods"].pop(methods[-1])
    with pytest.raises(ValueError, match="complete method set"):
        quality_table(changed)
