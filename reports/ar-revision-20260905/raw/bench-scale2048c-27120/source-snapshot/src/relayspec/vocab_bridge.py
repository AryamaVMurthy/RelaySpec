from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def decode_new_chunk(tokenizer: Any, token_ids: torch.Tensor) -> str:
    """Decode a 1-D tensor of token ids to text, dropping special tokens."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


def committed_suffix(
    target_block_ids: torch.Tensor,
    posterior: torch.Tensor,
    accepted: int,
) -> torch.Tensor:
    """The newly committed tokens this cycle, excluding the already-known anchor.

    `target_block_ids[:, 0]` is the anchor: the last token committed in a
    prior cycle, re-fed only so the target can predict what follows it. It
    must never be re-appended to the running committed sequence. The new
    content is the `accepted - 1` proposed tokens the target agreed with,
    plus the target's own correction token.
    """
    if accepted < 1 or accepted > target_block_ids.shape[1]:
        raise ValueError("accepted must be between 1 and the block length")
    return torch.cat(
        [target_block_ids[:, 1:accepted], posterior[:, accepted - 1 : accepted]],
        dim=1,
    )


def encode_new_chunk(
    tokenizer: Any, text: str, device: torch.device
) -> torch.LongTensor:
    """Encode a text chunk with no added special tokens, as a [1, N] tensor.

    Encoding an empty string must return an empty (not zero-dim) tensor, since
    S-A cross-family retargeting (see
    docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
    section 3.5b) can commit a chunk whose decoded text is empty (for
    example, a single partial-token correction with no other content).
    """
    if text == "":
        return torch.zeros((1, 0), dtype=torch.long, device=device)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return encoded["input_ids"].to(device)


def content_windows(rendered_text: str, *content_strings: str) -> list[tuple[int, int]]:
    """Character ranges where each of `content_strings` appears verbatim.

    Two models' chat templates render entirely different boilerplate around
    the same problem and solution text (a real system prompt, a date
    stamp, different role markers), so a rendered prompt is not the same
    string across tokenizers even though the content is. Character-span
    alignment (`align_positions`) is only meaningful inside the windows
    where the actual shared content sits; template boilerplate has no
    cross-model correspondence and must be excluded, not aligned to
    whatever happens to occupy the same character range in the other
    template. Raises if any content string cannot be found, since a
    template that quotes or otherwise transforms its input would silently
    produce an empty, wrong window instead of a clear failure.
    """
    windows: list[tuple[int, int]] = []
    for content in content_strings:
        start = rendered_text.find(content)
        if start >= 0:
            windows.append((start, start + len(content)))
            continue
        # Some chat templates strip leading/trailing whitespace from a
        # message's content before rendering it (a template-level `.strip()`
        # in the Jinja source, not something callers control), which is the
        # one transformation common enough to handle rather than fail on.
        # Anything else (re-wrapping, escaping) is a real template
        # incompatibility this alignment approach cannot handle, and must
        # still raise, with enough context to diagnose which content and
        # which template failed.
        stripped = content.strip()
        start = rendered_text.find(stripped) if stripped != content else -1
        if start >= 0:
            windows.append((start, start + len(stripped)))
            continue
        raise ValueError(
            "content string not found verbatim (or stripped) in rendered text: "
            f"content starts {content[:60]!r}, rendered text starts {rendered_text[:200]!r}"
        )
    return windows


def filter_offsets_to_windows(
    offsets: list[tuple[int, int]], windows: list[tuple[int, int]]
) -> tuple[list[int], list[tuple[int, int]]]:
    """Keep only the offsets fully inside one of `windows`.

    Returns the kept original indices alongside their offsets, so a caller
    can gather hidden states at the original (unfiltered) sequence
    positions after aligning within the filtered, content-only view.
    """
    kept_indices: list[int] = []
    kept_offsets: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if any(window_start <= start and end <= window_end for window_start, window_end in windows):
            kept_indices.append(index)
            kept_offsets.append((start, end))
    return kept_indices, kept_offsets


def canonicalize_offsets(
    offsets: list[tuple[int, int]], windows: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Remap offsets from one rendered string's coordinates into a shared
    content-only coordinate system, so two different renderings of the same
    content can be compared position by position.

    `windows` are this string's own content windows (`content_windows`),
    treated as if concatenated end to end starting at 0: the first window
    becomes `[0, len(window_0))`, the second `[len(window_0),
    len(window_0) + len(window_1))`, and so on. Calling this on both the
    source and target renderings' filtered offsets (`filter_offsets_to_windows`,
    with each string's own windows) puts them on the same coordinate axis,
    since both windows sequences cover the identical underlying problem and
    solution text in the identical order, only at different absolute
    positions in two differently templated strings. Every offset must lie
    inside exactly one window, which `filter_offsets_to_windows` already
    guarantees for its output.
    """
    window_canonical_starts: list[int] = []
    cursor = 0
    for window_start, window_end in windows:
        window_canonical_starts.append(cursor)
        cursor += window_end - window_start
    canonical: list[tuple[int, int]] = []
    for start, end in offsets:
        for (window_start, window_end), canonical_start in zip(
            windows, window_canonical_starts
        ):
            if window_start <= start and end <= window_end:
                shift = canonical_start - window_start
                canonical.append((start + shift, end + shift))
                break
        else:
            raise ValueError("offset does not lie inside any provided window")
    return canonical


def align_positions(
    source_offsets: list[tuple[int, int]],
    target_offsets: list[tuple[int, int]],
) -> list[int | None]:
    """For each target position, the source position it overlaps most.

    `source_offsets[i]` and `target_offsets[j]` are `(start, end)` character
    spans into the same underlying rendered text (see
    `docs/plans/2026-09-03-relayspec-cross-family-fix-plan.md`, Step 2).
    Two tokenizations of one string are both partitions of its character
    range, so "largest overlap" is a direct, deterministic alignment: no
    dynamic-programming or learned component is needed, only the two
    offset-mapping outputs `AutoTokenizer(..., return_offsets_mapping=True)`
    already provides. A target position whose span does not overlap any
    source span (only possible at special-token boundaries the two
    tokenizers do not agree on) aligns to `None` and must be dropped from
    the loss, not aligned to an arbitrary neighbor.
    """
    aligned: list[int | None] = []
    source_cursor = 0
    for target_start, target_end in target_offsets:
        if target_end <= target_start:
            aligned.append(None)
            continue
        while (
            source_cursor + 1 < len(source_offsets)
            and source_offsets[source_cursor][1] <= target_start
        ):
            source_cursor += 1
        best_index: int | None = None
        best_overlap = 0
        search_cursor = source_cursor
        while search_cursor < len(source_offsets):
            source_start, source_end = source_offsets[search_cursor]
            if source_start >= target_end:
                break
            overlap = min(source_end, target_end) - max(source_start, target_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = search_cursor
            search_cursor += 1
        aligned.append(best_index)
    return aligned


def build_vocab_intersection(source_tokenizer: Any, target_tokenizer: Any) -> dict[int, int]:
    """Map every source token id whose text is exactly one target token too.

    S-A (`encode_new_chunk` after `decode_new_chunk`) always round-trips
    through text and is correct but re-tokenizes even when the two
    tokenizers would have produced identical output. This is S-B from
    docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
    section 3.5b: build the injective partial map once per tokenizer pair
    (a one-time, model-free, CPU-only computation over both vocabularies),
    then use it to skip the round trip for any run of tokens it fully
    covers, falling back to S-A only where it does not. Correctness is
    unaffected either way, since a covered token is defined as one whose
    text a full S-A round trip would have produced anyway.
    """
    target_text_to_id: dict[str, int] = {}
    for target_id in range(len(target_tokenizer)):
        text = target_tokenizer.decode([target_id], skip_special_tokens=False)
        target_text_to_id.setdefault(text, target_id)
    intersection: dict[int, int] = {}
    for source_id in range(len(source_tokenizer)):
        text = source_tokenizer.decode([source_id], skip_special_tokens=False)
        target_id = target_text_to_id.get(text)
        if target_id is None:
            continue
        if target_tokenizer.decode([target_id], skip_special_tokens=False) != text:
            continue
        if source_tokenizer.decode([source_id], skip_special_tokens=False) != text:
            continue
        intersection[source_id] = target_id
    return intersection


def save_vocab_intersection(intersection: dict[int, int], path: Path) -> None:
    path.write_text(
        json.dumps({str(k): v for k, v in intersection.items()}), encoding="utf-8"
    )


def load_vocab_intersection(path: Path) -> dict[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def bridge_encode_new_chunk(
    source_ids: torch.Tensor,
    text: str,
    target_tokenizer: Any,
    device: torch.device,
    intersection: dict[int, int],
) -> torch.LongTensor:
    """Re-tokenize a just-proposed chunk into the target vocabulary.

    `source_ids` is the proposer's own output for this chunk, still in
    source-vocabulary ids, and `text` is its decoded form (S-A already
    computed both). If every id in the chunk is covered by `intersection`,
    the mapped target ids are used directly with no re-encoding. Otherwise
    this falls back to the ordinary S-A path (`encode_new_chunk` on the
    decoded text), so a partial-coverage chunk never mixes the two: mixing
    would require deciding, token by token, whether the target tokenizer
    would have produced the same boundary, which the intersection map does
    not by itself guarantee for a multi-token run.
    """
    ids = source_ids.tolist()
    mapped = [intersection.get(i) for i in ids]
    if all(value is not None for value in mapped):
        return torch.tensor([mapped], dtype=torch.long, device=device)
    return encode_new_chunk(target_tokenizer, text, device)
