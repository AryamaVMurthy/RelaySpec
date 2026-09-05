from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class _FakeTokenizer:
    """Maps single characters to token ids one-for-one, for isolated tests."""

    def __init__(self, size: int = 26) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(ord("a") + i) for i in ids)

    def __call__(
        self, text: str, add_special_tokens: bool = False, return_tensors=None
    ):
        ids = [ord(ch) - ord("a") for ch in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _MappedTokenizer:
    """A tiny tokenizer whose id-to-text map is given explicitly, so tests
    can control exactly which ids overlap with another tokenizer's."""

    def __init__(self, id_to_text: dict[int, str]) -> None:
        self._id_to_text = id_to_text
        self._text_to_id = {v: k for k, v in id_to_text.items()}

    def __len__(self) -> int:
        return len(self._id_to_text)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self._id_to_text[i] for i in ids)

    def __call__(
        self, text: str, add_special_tokens: bool = False, return_tensors=None
    ):
        ids = [self._text_to_id[ch] for ch in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_decode_new_chunk_round_trips_through_fake_tokenizer() -> None:
    from relayspec.vocab_bridge import decode_new_chunk

    tokenizer = _FakeTokenizer()
    assert decode_new_chunk(tokenizer, torch.tensor([0, 1, 2])) == "abc"


def test_encode_new_chunk_handles_empty_string_without_a_zero_dim_tensor() -> None:
    from relayspec.vocab_bridge import encode_new_chunk

    tokenizer = _FakeTokenizer()
    result = encode_new_chunk(tokenizer, "", torch.device("cpu"))
    assert result.shape == (1, 0)


def test_encode_new_chunk_encodes_nonempty_text() -> None:
    from relayspec.vocab_bridge import encode_new_chunk

    tokenizer = _FakeTokenizer()
    result = encode_new_chunk(tokenizer, "ab", torch.device("cpu"))
    assert result.tolist() == [[0, 1]]


def test_committed_suffix_excludes_the_anchor_on_full_acceptance() -> None:
    from relayspec.vocab_bridge import committed_suffix

    target_block_ids = torch.tensor([[10, 11, 12, 13]])
    posterior = torch.tensor([[11, 12, 13, 14]])
    result = committed_suffix(target_block_ids, posterior, accepted=4)
    assert result.tolist() == [[11, 12, 13, 14]]


def test_committed_suffix_excludes_the_anchor_on_zero_proposal_agreement() -> None:
    from relayspec.vocab_bridge import committed_suffix

    target_block_ids = torch.tensor([[10, 11, 12]])
    posterior = torch.tensor([[99, 98, 97]])
    result = committed_suffix(target_block_ids, posterior, accepted=1)
    assert result.tolist() == [[99]]
    assert 10 not in result.tolist()[0]


def test_committed_suffix_never_reproduces_the_anchor_value() -> None:
    from relayspec.vocab_bridge import committed_suffix

    anchor = 42
    target_block_ids = torch.tensor([[anchor, 1, 2, 3, 4]])
    posterior = torch.tensor([[1, 2, 3, 4, 5]])
    for accepted in range(1, 6):
        result = committed_suffix(target_block_ids, posterior, accepted)
        assert anchor not in result.tolist()[0]


def test_committed_suffix_rejects_out_of_range_accepted() -> None:
    from relayspec.vocab_bridge import committed_suffix

    target_block_ids = torch.tensor([[1, 2, 3]])
    posterior = torch.tensor([[2, 3, 4]])
    with pytest.raises(ValueError):
        committed_suffix(target_block_ids, posterior, accepted=0)
    with pytest.raises(ValueError):
        committed_suffix(target_block_ids, posterior, accepted=4)


def test_content_windows_finds_verbatim_substrings() -> None:
    from relayspec.vocab_bridge import content_windows

    rendered = "<sys>boilerplate<user>hello world<assist>the answer<end>"
    windows = content_windows(rendered, "hello world", "the answer")
    assert rendered[windows[0][0] : windows[0][1]] == "hello world"
    assert rendered[windows[1][0] : windows[1][1]] == "the answer"


def test_content_windows_raises_when_content_missing() -> None:
    from relayspec.vocab_bridge import content_windows

    with pytest.raises(ValueError):
        content_windows("<sys>only boilerplate<end>", "not present")


def test_content_windows_falls_back_to_stripped_content() -> None:
    from relayspec.vocab_bridge import content_windows

    # A template that strips leading/trailing whitespace from a message's
    # content before rendering it: "  hello world  " never appears
    # verbatim, but its stripped form does.
    rendered = "<sys>boilerplate<user>hello world<assist>the answer<end>"
    windows = content_windows(rendered, "  hello world  ", "the answer")
    assert rendered[windows[0][0] : windows[0][1]] == "hello world"


def test_content_windows_error_message_shows_context() -> None:
    from relayspec.vocab_bridge import content_windows

    with pytest.raises(ValueError, match="content starts"):
        content_windows("<sys>only boilerplate<end>", "not present anywhere")


def test_filter_offsets_to_windows_keeps_only_covered_offsets() -> None:
    from relayspec.vocab_bridge import filter_offsets_to_windows

    # window [5, 15) is the only allowed region
    offsets = [(0, 3), (5, 8), (8, 15), (14, 17), (20, 22)]
    windows = [(5, 15)]
    kept_indices, kept_offsets = filter_offsets_to_windows(offsets, windows)
    # (0,3) is outside, (14,17) crosses the boundary and is excluded,
    # (20,22) is outside. Only fully-contained spans survive.
    assert kept_indices == [1, 2]
    assert kept_offsets == [(5, 8), (8, 15)]


def test_filter_offsets_to_windows_supports_multiple_disjoint_windows() -> None:
    from relayspec.vocab_bridge import filter_offsets_to_windows

    offsets = [(0, 2), (10, 12), (20, 22), (30, 32)]
    windows = [(0, 5), (20, 25)]
    kept_indices, kept_offsets = filter_offsets_to_windows(offsets, windows)
    assert kept_indices == [0, 2]
    assert kept_offsets == [(0, 2), (20, 22)]


def test_canonicalize_offsets_puts_two_windows_on_shared_axis() -> None:
    from relayspec.vocab_bridge import canonicalize_offsets

    # Two windows starting at very different absolute positions (as if in
    # two differently-templated renderings), each 5 and 4 characters long.
    windows = [(100, 105), (200, 204)]
    offsets = [(100, 102), (102, 105), (200, 202), (202, 204)]
    result = canonicalize_offsets(offsets, windows)
    # window 0 -> canonical [0, 5), window 1 -> canonical [5, 9)
    assert result == [(0, 2), (2, 5), (5, 7), (7, 9)]


def test_canonicalize_offsets_makes_two_renderings_comparable() -> None:
    from relayspec.vocab_bridge import canonicalize_offsets

    # Same content ("hi" + "bye"), rendered with different boilerplate
    # lengths in two fake templates: source has a short prefix, target a
    # long one, but the canonical form should be identical.
    source_windows = [(3, 5), (10, 13)]  # "hi" at 3, "bye" at 10
    target_windows = [(50, 52), (80, 83)]  # "hi" at 50, "bye" at 80
    source_offsets = [(3, 5), (10, 13)]
    target_offsets = [(50, 52), (80, 83)]
    assert canonicalize_offsets(source_offsets, source_windows) == canonicalize_offsets(
        target_offsets, target_windows
    )


def test_align_positions_identity_when_tokenizations_match() -> None:
    from relayspec.vocab_bridge import align_positions

    offsets = [(0, 2), (2, 5), (5, 8)]
    assert align_positions(offsets, offsets) == [0, 1, 2]


def test_align_positions_picks_larger_overlap_when_source_is_finer() -> None:
    from relayspec.vocab_bridge import align_positions

    # source splits "hello" into "he" + "llo", target keeps it as one token
    source_offsets = [(0, 2), (2, 5)]
    target_offsets = [(0, 5)]
    assert align_positions(source_offsets, target_offsets) == [1]


def test_align_positions_picks_larger_overlap_when_target_is_finer() -> None:
    from relayspec.vocab_bridge import align_positions

    # source keeps "hello" as one token, target splits it "hel" + "lo"
    source_offsets = [(0, 5)]
    target_offsets = [(0, 3), (3, 5)]
    assert align_positions(source_offsets, target_offsets) == [0, 0]


def test_align_positions_handles_multiple_target_positions_in_order() -> None:
    from relayspec.vocab_bridge import align_positions

    # "the cat sat" tokenized differently: source word-level, target char-pair
    source_offsets = [(0, 3), (4, 7), (8, 11)]  # "the", "cat", "sat"
    target_offsets = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 11)]
    result = align_positions(source_offsets, target_offsets)
    assert result == [0, 0, 1, 1, 2, 2]


def test_align_positions_returns_none_for_empty_span() -> None:
    from relayspec.vocab_bridge import align_positions

    source_offsets = [(0, 3)]
    target_offsets = [(0, 0), (0, 3)]
    result = align_positions(source_offsets, target_offsets)
    assert result[0] is None
    assert result[1] == 0


def test_align_positions_returns_none_when_no_source_overlap() -> None:
    from relayspec.vocab_bridge import align_positions

    source_offsets = [(0, 3)]
    target_offsets = [(0, 3), (3, 6)]
    result = align_positions(source_offsets, target_offsets)
    assert result[0] == 0
    assert result[1] is None


def test_build_vocab_intersection_finds_only_shared_text() -> None:
    from relayspec.vocab_bridge import build_vocab_intersection

    source = _MappedTokenizer({0: "a", 1: "b", 2: "q"})
    target = _MappedTokenizer({0: "b", 1: "a", 2: "z"})
    intersection = build_vocab_intersection(source, target)
    assert intersection == {0: 1, 1: 0}
    assert 2 not in intersection


def test_save_and_load_vocab_intersection_round_trips(tmp_path) -> None:
    from relayspec.vocab_bridge import load_vocab_intersection, save_vocab_intersection

    original = {5: 9, 12: 1}
    path = tmp_path / "intersection.json"
    save_vocab_intersection(original, path)
    assert load_vocab_intersection(path) == original


def test_bridge_encode_new_chunk_uses_intersection_when_fully_covered() -> None:
    from relayspec.vocab_bridge import bridge_encode_new_chunk

    target = _MappedTokenizer({0: "b", 1: "a"})
    intersection = {0: 1, 1: 0}
    source_ids = torch.tensor([0, 1])
    result = bridge_encode_new_chunk(
        source_ids, "ba", target, torch.device("cpu"), intersection
    )
    assert result.tolist() == [[1, 0]]


def test_bridge_encode_new_chunk_falls_back_when_not_covered() -> None:
    from relayspec.vocab_bridge import bridge_encode_new_chunk

    target = _MappedTokenizer({0: "b", 1: "a"})
    intersection = {0: 1}
    source_ids = torch.tensor([0, 2])
    result = bridge_encode_new_chunk(
        source_ids, "ba", target, torch.device("cpu"), intersection
    )
    # source id 2 is uncovered, so this falls back to re-encoding "ba"
    # through the target tokenizer directly: "b" -> 0, "a" -> 1.
    assert result.tolist() == [[0, 1]]
