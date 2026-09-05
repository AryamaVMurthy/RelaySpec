from relayspec.scaling_data import OverlapIndex, stratified_order


def test_lexical_audit_detects_exact_and_near_without_common_phrase_false_positive():
    index = OverlapIndex()
    text = "Find all positive integer solutions to the equation x squared plus y squared equals 25 and justify each solution carefully."
    index.add(text)
    assert index.match("  " + text.upper()) == "exact"
    assert index.match(text + " Explain carefully.") == "near"
    assert index.match("Find the number of students in the classroom.") is None


def test_nested_order_is_reproducible_and_stratified():
    rows = [
        dict(source=s, problem=f"{s}-{i}", solution="answer")
        for s, n in [("a", 60), ("b", 40)]
        for i in range(n)
    ]
    first = list(stratified_order(rows, 1729))
    assert first == list(stratified_order(list(reversed(rows)), 1729))
    assert sum(r["source"] == "a" for r in first[:10]) == 6
    assert len({r["problem"] for r in first}) == 100
