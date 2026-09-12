"""Conservative source-label anchors for target-token conditioning.

No token-ID lookup bridge. An eligible target anchor is a complete, round-trip
stable text prefix under the source tokenizer's native normalization. Source
prefix token IDs must match the full-sequence token prefix exactly, so this does
not permit future text to influence an anchor. Source labels are tokens following
that prefix. Context visibility must remain target positions strictly BEFORE
this target anchor: the clean source anchor is supplied as a token embedding.
"""
import json


def complete_unicode_prefix(target_ids, tokenizer, decoded):
    """Handle only a byte-level rollout capped inside its final UTF-8 character.

    The target rollout is retained verbatim. Source supervision ends at the last
    whole target token before that partial character; malformed interior bytes
    and literal replacement characters remain errors.
    """
    failure = 'Lossy Unicode decode: cannot establish exact prefixes'
    decoder = getattr(getattr(tokenizer, 'backend_tokenizer', None), 'decoder', None)
    if decoder is None or json.loads(decoder.__getstate__()).get('type') != 'ByteLevel':
        raise ValueError(failure)
    # Invert the byte-level BPE alphabet (printable bytes retain their codepoint;
    # other bytes map, in byte order, to codepoints starting at256).
    kept = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    alphabet = {chr(b):b for b in kept}
    for i,b in enumerate(b for b in range(256) if b not in kept):
        alphabet[chr(256+i)] = b
    try:
        pieces = [bytes(alphabet[c] for c in piece)
                  for piece in tokenizer.convert_ids_to_tokens(target_ids)]
    except KeyError as error:
        raise ValueError(failure) from error
    raw = b''.join(pieces)
    if raw.decode('utf8',errors='replace') != decoded:
        raise ValueError(failure)
    try:
        raw.decode('utf8',errors='strict')
    except UnicodeDecodeError as error:
        if error.reason != 'unexpected end of data' or error.end != len(raw):
            raise ValueError(failure) from error
        valid_bytes = error.start
    else:
        raise ValueError(failure)  # Literal U+FFFD is not an incomplete byte tail.
    cutoff = 0;offset = 0
    for piece in pieces:
        if offset + len(piece) > valid_bytes:
            break
        offset += len(piece);cutoff += 1
    # A token can finish one character and begin another. Back up to a whole
    # character as well as a whole token, without synthesizing replacement text.
    while cutoff:
        try:
            raw[:offset].decode('utf8',errors='strict')
            break
        except UnicodeDecodeError:
            cutoff -= 1;offset -= len(pieces[cutoff])
    prefix = tokenizer.decode(target_ids[:cutoff],skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
    if '\ufffd' in prefix or prefix != raw[:offset].decode('utf8',errors='strict'):
        raise ValueError(failure)
    return prefix,dict(aligned_target_tokens=cutoff,
                       excluded_trailing_target_tokens=len(target_ids)-cutoff,
                       reason='rollout ends inside a UTF-8 character; target IDs unchanged')


def aligned_blocks(target_ids, prompt_length, source_tokenizer, target_tokenizer,
                   block_size=16, candidate_positions=None):
    if not 0 < prompt_length < len(target_ids):
        raise ValueError('Need a nonempty prompt and generated continuation')
    text = target_tokenizer.decode(target_ids, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
    unicode_tail = None
    if '\ufffd' in text:
        text,unicode_tail = complete_unicode_prefix(target_ids,target_tokenizer,text)
        if unicode_tail['aligned_target_tokens'] <= prompt_length:
            raise ValueError('No complete generated Unicode prefix')
    source = source_tokenizer(text, add_special_tokens=False,
                              return_offsets_mapping=True)
    source_ids = source['input_ids']
    normalizer = getattr(getattr(source_tokenizer, 'backend_tokenizer', None),
                         'normalizer', None)
    # Candidate character boundaries only. Offset overlap/partial bytes are
    # not enough: prefix encoding and decoding are both checked below.
    ends = {}
    for j, (start, end) in enumerate(source['offset_mapping']):
        if end > start:
            ends[end] = j
    blocks = []
    reasons = {'not_text_prefix': 0, 'no_shared_boundary': 0,
               'unstable_source_prefix': 0, 'no_labels': 0}
    # Paired-feature sampling is fixed before alignment. Checking only that
    # subset is equivalent to checking everything and then filtering blocks;
    # retain target order and ignore out-of-range candidates as filtering does.
    candidates = (range(prompt_length, len(target_ids) - 1)
                  if candidate_positions is None else
                  sorted(k for k in set(candidate_positions)
                         if prompt_length <= k < len(target_ids) - 1))
    for k in candidates:
        prefix = target_tokenizer.decode(target_ids[:k+1],
                    skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if '\ufffd' in prefix or not text.startswith(prefix):
            reasons['not_text_prefix'] += 1
            continue
        j = ends.get(len(prefix))
        if j is None:
            reasons['no_shared_boundary'] += 1
            continue
        encoded_prefix = source_tokenizer.encode(prefix, add_special_tokens=False)
        source_prefix = normalizer.normalize_str(prefix) if normalizer is not None else prefix
        if encoded_prefix != source_ids[:j+1] or source_tokenizer.decode(
                encoded_prefix, skip_special_tokens=False,
                clean_up_tokenization_spaces=False) != source_prefix:
            reasons['unstable_source_prefix'] += 1
            continue
        labels = source_ids[j:j+block_size]
        if len(labels) < 2:
            reasons['no_labels'] += 1
            continue
        blocks.append({'target_anchor': k, 'source_anchor': j,
                       'prefix_characters': len(prefix), 'source_labels': labels,
                       'context_exclusive_end': k})
    return {'source_ids': source_ids, 'blocks': blocks, 'rejected': reasons,
            **({'unicode_tail':unicode_tail} if unicode_tail else {}),
            'context_position_units': 'target tokens',
            'label_position_units': 'source tokens',
            'scope': 'AUF source-token prefix extension; not target-token AUF'}


def paired_from_cache(target_ids, prompt_length, source_tokenizer, target_tokenizer,
                      selected, generated):
    """Extend verified generated-prefix labels with sampled prompt positions.

    Caller must verify the cached manifest/hash and row identity. Re-encoding the
    full source sequence below detects a mismatched tokenizer or sequence. Every
    generated anchor already passed the identical prefix checks in aligned_blocks.
    """
    selected=set(selected)
    prompt=aligned_blocks(target_ids,1,source_tokenizer,target_tokenizer,
                          candidate_positions=[k for k in selected if k<prompt_length])
    if prompt['source_ids']!=generated['source_ids']:
        raise ValueError('Cached alignment has a different source sequence')
    if any(b['target_anchor']<prompt_length for b in generated['blocks']):
        raise ValueError('Generated alignment cache contains prompt anchors')
    blocks=prompt['blocks']+[b for b in generated['blocks'] if b['target_anchor'] in selected]
    return {'source_ids':prompt['source_ids'],'blocks':blocks}
