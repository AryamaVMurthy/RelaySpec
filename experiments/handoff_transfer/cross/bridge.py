"""Stateful full-prefix anchors and text proposals; target verification is separate."""
class TextBridge:
    def __init__(self,source,target):
        self.source=source;self.target=target;self.histories={}
    def anchor(self,key,positions,tokens,bonus):
        if not positions or len(positions)!=len(tokens):raise ValueError('Missing target context')
        if positions[0]==0:self.histories[key]=[]
        history=self.histories.get(key)
        if history is None:raise ValueError('Missing request prefix')
        for position,token in zip(positions,tokens):
            if position>len(history):raise ValueError('Gap in committed target prefix')
            if position<len(history):
                if history[position]!=token:raise ValueError('Previously committed target token changed')
            else:history.append(token)
        bonus_position=positions[-1]+1
        if bonus_position<len(history):
            if history[bonus_position]!=bonus:raise ValueError('Bonus token disagrees with prefix')
            del history[bonus_position+1:]
        else:history.append(bonus)
        text=self.target.decode(history,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        ids=self.source.encode(text,add_special_tokens=False)
        if not ids:raise ValueError('Target prefix has no source anchor')
        return ids[-1]
    def proposals(self,ids,count):
        text=self.source.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        target_ids=self.target.encode(text,add_special_tokens=False)[:count]
        # Fixed-width vLLM proposal buffers: remaining positions explicitly
        # propose target EOS, which still undergoes ordinary target verification.
        if self.target.eos_token_id is None:raise ValueError('Target EOS required for proposal padding')
        return target_ids+[self.target.eos_token_id]*(count-len(target_ids))
