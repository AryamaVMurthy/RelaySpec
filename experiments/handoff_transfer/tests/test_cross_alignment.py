from experiments.handoff_transfer.cross.alignment import aligned_blocks
import pytest

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
