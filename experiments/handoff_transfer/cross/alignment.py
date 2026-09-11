"""Conservative source-label anchors for target-token conditioning.

No token-ID lookup bridge. An eligible target anchor is a complete, round-trip
stable text prefix in both tokenizers. Source labels are source tokens following
that prefix. Context visibility must remain target positions strictly BEFORE
this target anchor: the clean source anchor is supplied as a token embedding.
"""


def aligned_blocks(target_ids, prompt_length, source_tokenizer, target_tokenizer,
                   block_size=16):
    if not 0 < prompt_length < len(target_ids):
        raise ValueError('Need a nonempty prompt and generated continuation')
    text = target_tokenizer.decode(target_ids, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
    if '\ufffd' in text:
        raise ValueError('Lossy Unicode decode: cannot establish exact prefixes')
    source = source_tokenizer(text, add_special_tokens=False,
                              return_offsets_mapping=True)
    source_ids = source['input_ids']
    # Candidate character boundaries only. Offset overlap/partial bytes are
    # not enough: prefix encoding and decoding are both checked below.
    ends = {}
    for j, (start, end) in enumerate(source['offset_mapping']):
        if end > start:
            ends[end] = j
    blocks = []
    reasons = {'not_text_prefix': 0, 'no_shared_boundary': 0,
               'unstable_source_prefix': 0, 'no_labels': 0}
    for k in range(prompt_length, len(target_ids) - 1):
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
        if encoded_prefix != source_ids[:j+1] or source_tokenizer.decode(
                encoded_prefix, skip_special_tokens=False,
                clean_up_tokenization_spaces=False) != prefix:
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
            'context_position_units': 'target tokens',
            'label_position_units': 'source tokens',
            'scope': 'AUF source-token prefix extension; not target-token AUF'}
