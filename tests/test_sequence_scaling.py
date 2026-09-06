import pytest

from relayspec.sequence_scaling import token_ids_digest, validate_exact_input


def test_exact_input_rejects_truncation_mutation_and_window_overflow():
    r = {
        "input_ids": [1, 2, 3],
        "context_tokens": 3,
        "input_ids_sha256": token_ids_digest([1, 2, 3]),
    }
    assert validate_exact_input(r, vocab_size=10, max_positions=100, output_cap=16) == [
        1,
        2,
        3,
    ]
    with pytest.raises(ValueError, match="position"):
        validate_exact_input(r, vocab_size=10, max_positions=40, output_cap=16)
    with pytest.raises(ValueError, match="hash"):
        validate_exact_input(
            {**r, "input_ids": [1, 2, 4]},
            vocab_size=10,
            max_positions=100,
            output_cap=16,
        )
    with pytest.raises(ValueError, match="length"):
        validate_exact_input(
            {**r, "input_ids": [1, 2]}, vocab_size=10, max_positions=100, output_cap=16
        )
