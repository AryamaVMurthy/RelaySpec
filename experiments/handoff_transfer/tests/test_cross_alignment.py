from experiments.handoff_transfer.cross.alignment import aligned_blocks
from experiments.handoff_transfer.cross import alignment
import pytest
import unicodedata
from types import SimpleNamespace

class Tokenizer:
    def __init__(self, pieces):
        self.pieces=pieces
    def __call__(self,text,**kwargs):
        ids=[];offsets=[];i=0
        while i<len(text):
            choices=[(len(p),j,p) for j,p in enumerate(self.pieces) if text.startswith(p,i)]
            _,j,p=max(choices)
            ids.append(j);offsets.append((i,i+len(p)));i+=len(p)
        return {'input_ids':ids,'offset_mapping':offsets}
    def encode(self,text,**kwargs):return self(text)['input_ids']
    def decode(self,ids,**kwargs):return ''.join(self.pieces[i] for i in ids)

def test_distinct_vocab_ids_align_text_and_keep_future_context_hidden():
    target=Tokenizer(list('abcdefghij'))
    source=Tokenizer(['ab','cd','ef','gh','ij']+list('abcdefghij'))
    ids=target.encode('abcdefghij')
    result=aligned_blocks(ids,1,source,target,block_size=3)
    assert [(b['target_anchor'],b['source_anchor']) for b in result['blocks']]==[(1,0),(3,1),(5,2),(7,3)]
    for b in result['blocks']:
        assert b['context_exclusive_end']==b['target_anchor']
        assert source.decode(b['source_labels'])=='abcdefghij'[b['target_anchor']-1:b['target_anchor']-1+6]
        assert source.decode(result['source_ids'][:b['source_anchor']+1])==target.decode(ids[:b['target_anchor']+1])
    assert result['rejected']['no_shared_boundary']>0

def test_lossy_unicode_is_rejected():
    tok=Tokenizer(list('ab\ufffdc'))
    with pytest.raises(ValueError,match='Lossy Unicode'):
        aligned_blocks(tok.encode('ab\ufffdc'),1,tok,tok)

@pytest.mark.parametrize('selected', [[], [7, 1, 7, 2, -1, 100], range(0, 10, 2), range(10)])
def test_selected_alignment_preserves_full_then_filter(selected):
    target=Tokenizer(list('abcdefghij'))
    source=Tokenizer(['ab','cd','ef','gh','ij']+list('abcdefghij'))
    ids=target.encode('abcdefghij')
    full=aligned_blocks(ids,1,source,target,block_size=3)
    subset=aligned_blocks(ids,1,source,target,block_size=3,candidate_positions=selected)
    assert subset['source_ids']==full['source_ids']
    assert subset['blocks']==[b for b in full['blocks'] if b['target_anchor'] in selected]

def test_selected_alignment_retains_unicode_rejection():
    tok=Tokenizer(list('ab\ufffdc'))
    with pytest.raises(ValueError,match='Lossy Unicode'):
        aligned_blocks(tok.encode('ab\ufffdc'),1,tok,tok,candidate_positions=[])


def test_native_source_normalization_keeps_causal_token_prefixes():
    class NFCTokenizer(Tokenizer):
        backend_tokenizer=SimpleNamespace(normalizer=SimpleNamespace(
            normalize_str=lambda text:unicodedata.normalize('NFC',text)))
        def __call__(self,text,**kwargs):
            # U+2001 -> U+2003 has equal character length and reproduces the
            # real Qwen/Llama failed prompt without downloading tokenizers.
            return super().__call__(unicodedata.normalize('NFC',text),**kwargs)
    target=Tokenizer(list('abcde\u2001'))
    source=NFCTokenizer(list('abcde\u2003'))
    ids=target.encode('a\u2001bcde')
    result=aligned_blocks(ids,2,source,target,block_size=3)
    assert len(result['blocks'])==3
    for block in result['blocks']:
        k,j=block['target_anchor'],block['source_anchor']
        prefix=target.decode(ids[:k+1])
        assert source.encode(prefix)==result['source_ids'][:j+1]
        assert source.decode(result['source_ids'][:j+1])==unicodedata.normalize('NFC',prefix)
        assert block['context_exclusive_end']==k


def test_decoder_text_changes_without_declared_normalization_stay_rejected():
    class LossyTokenizer(Tokenizer):
        def decode(self,ids,**kwargs):return super().decode(ids,**kwargs).replace('b','a')
    target=Tokenizer(list('abcde'));source=LossyTokenizer(list('abcde'))
    result=aligned_blocks(target.encode('abcde'),2,source,target)
    assert result['blocks']==[]


@pytest.mark.parametrize('selected', [[], range(10), [0,1,3,4,7,7,9,100,-1]])
def test_cached_generated_alignment_preserves_paired_sampling(selected):
    target=Tokenizer(list('abcdefghij'))
    source=Tokenizer(['ab','cd','ef','gh','ij']+list('abcdefghij'))
    ids=target.encode('abcdefghij');prompt_length=4
    cached=aligned_blocks(ids,prompt_length,source,target)
    expected=aligned_blocks(ids,1,source,target,candidate_positions=selected)
    assert hasattr(alignment,'paired_from_cache'), 'Missing reuse of audited generated-prefix alignment'
    actual=alignment.paired_from_cache(ids,prompt_length,source,target,selected,cached)
    assert actual['source_ids']==expected['source_ids']
    assert actual['blocks']==expected['blocks']


def test_paired_cache_rejects_different_source_sequence():
    target=Tokenizer(list('abcdefghij'));source=Tokenizer(list('abcdefghij'))
    ids=target.encode('abcdefghij');cached=aligned_blocks(ids,4,source,target)
    cached['source_ids'][0]=9
    assert hasattr(alignment,'paired_from_cache'), 'Missing validated cache reuse'
    with pytest.raises(ValueError,match='source sequence'):
        alignment.paired_from_cache(ids,4,source,target,range(10),cached)


class ByteTokenizer(Tokenizer):
    backend_tokenizer=SimpleNamespace(decoder=SimpleNamespace(__getstate__=lambda:b'{"type":"ByteLevel"}'))
    def __init__(self):
        super().__init__(list('abcdef')+['â','Ī','ļ'])
    def convert_ids_to_tokens(self,ids):return [self.pieces[i] for i in ids]
    def decode(self,ids,**kwargs):
        values={'â':0xe2,'Ī':0x88,'ļ':0x9a}
        raw=bytes(values.get(self.pieces[i],ord(self.pieces[i])) for i in ids)
        return raw.decode('utf8',errors='replace')


def test_incomplete_utf8_tail_excludes_only_unalignable_target_tokens():
    target=ByteTokenizer();source=Tokenizer(list('abcdef'))
    ids=[0,1,2,3,4,5,6,7]  # abcdef plus two bytes of the three-byte sqrt sign
    result=aligned_blocks(ids,1,source,target,block_size=3)
    assert result['source_ids']==source.encode('abcdef')
    assert result['unicode_tail']['excluded_trailing_target_tokens']==2
    assert result['unicode_tail']['aligned_target_tokens']==6
    assert all(b['target_anchor']<6 and b['context_exclusive_end']==b['target_anchor'] for b in result['blocks'])
    assert result['blocks']==aligned_blocks(ids[:6],1,source,target,block_size=3)['blocks']
    assert ids==[0,1,2,3,4,5,6,7]  # The full target rollout is never changed.


def test_interior_invalid_utf8_still_rejected():
    target=ByteTokenizer();source=Tokenizer(list('abcdef'))
    with pytest.raises(ValueError,match='Lossy Unicode'):
        aligned_blocks([0,1,6,2,3,4],1,source,target)
