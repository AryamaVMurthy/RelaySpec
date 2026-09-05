from relayspec.confirmation_data import CandidateOverlapIndex
from relayspec.scaling_data import OverlapIndex


def test_reverse_exclusion_matches_individual_audits_including_short_duplicates():
    base = "Find all positive integer solutions to x squared plus y squared equals 25 and justify each solution carefully."
    candidates = [
        base,
        base + " Explain carefully.",
        "  SHORT text",
        "short text",
        "Unrelated question",
    ]
    index = CandidateOverlapIndex(candidates)
    for query in [
        base.upper(),
        base + " Explain carefully.",
        "short text",
        "Another question",
    ]:
        expected = {}
        for i, candidate in enumerate(candidates):
            one = OverlapIndex()
            one.add(candidate)
            reason = one.match(query)
            if reason:
                expected[i] = reason
        assert index.matches(query) == expected
    assert set(index.matches(base)) == {0, 1}
    assert index.matches("short text") == {2: "exact", 3: "exact"}
