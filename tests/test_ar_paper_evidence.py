import pytest

from relayspec.ar_paper_evidence import summarize


def rows():
    return [
        {
            "problem_id": f"gsm8k/{i}",
            "repetition": 0,
            "benchmark": "gsm8k",
            "method": m,
            "request_seconds": seconds,
            "output_tokens": tokens,
            "output_hash": m,
            "acceptance_lengths": [1, 2],
            "correct": True,
        }
        for i in range(2)
        for m, seconds, tokens in [("native_ar", 4, 20), ("relay", 2, 15)]
    ]


def test_throughput_and_time_ratios_differ_with_output_lengths():
    q = summarize(rows(), samples=20)["methods"]["relay"]
    assert q["throughput_ratio"] == 1.5
    assert q["request_time_ratio"] == 2.0
    assert q["throughput_ci95"] == [1.5, 1.5]


def test_ar_baseline_must_be_measured_in_the_same_run():
    with pytest.raises(ValueError, match="reference was not measured"):
        summarize([r for r in rows() if r["method"] == "relay"])


def test_incomplete_and_duplicate_pairs_fail():
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(rows()[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rows() + rows()[:1])


def test_two_chat_turns_are_one_bootstrap_unit():
    data = rows()
    for r in data:
        r["benchmark"] = "mtbench"
        r["problem_id"] = "mtbench/0/turn" + r["problem_id"].split("/")[-1]
        del r["correct"]
    result = summarize(data, samples=20)
    assert result["requests"] == 2
    assert result["clusters"] == 1
