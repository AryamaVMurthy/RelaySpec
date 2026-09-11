"""DFlash training with target-position contexts and source-vocabulary blocks.

This is an explicit heterogeneous-vocabulary extension. The CE/AUF objective
operates on source tokens, not on the target's accepted-token count.
"""
import torch
from torch.utils.checkpoint import checkpoint


def sample_aligned_anchors(valid, limit):
    counts=valid.sum(dim=1)
    width=min(limit,int(counts.max().item()))
    if width==0 or (counts==0).any():raise ValueError('Every record needs an aligned supervised block')
    scores=torch.rand(valid.shape,device=valid.device).masked_fill(~valid,2.)
    selected=scores.argsort(dim=1)[:,:width]
    keep=torch.arange(width,device=valid.device)[None,:]<counts[:,None].clamp(max=width)
    sentinel=valid.shape[1]
    selected=torch.where(keep,selected,sentinel).sort(dim=1).values
    keep=selected<sentinel
    return selected.masked_fill(~keep,0),keep


def gather_source_blocks(blocks,validity,anchors,keep):
    if blocks.shape!=validity.shape:raise ValueError('Label/validity shapes differ')
    index=anchors[:,:,None].expand(-1,-1,blocks.shape[-1])
    labels=blocks.gather(1,index)
    weights=validity.gather(1,index).float()*keep[:,:,None]
    weights[:,:,0]=0 # clean anchor, never a prediction target
    return labels,weights


def cross_online_class(online_base):
    class CrossOnline(online_base):
        def _sample_anchor_positions(self,seq_len,loss_mask,device,max_valid_anchors=None):
            # Eligibility was proven from both tokenizers. Adjacent target
            # positions need not align to adjacent source positions.
            return sample_aligned_anchors(loss_mask[:,:seq_len]>.5,self.num_anchors)

        def forward(self,input_ids,hidden_states,loss_mask,source_blocks,source_validity,
                    collect_detailed_metrics=False):
            if collect_detailed_metrics:raise ValueError('Cross-family target-token diagnostics are not defined')
            if input_ids.shape!=source_blocks.shape[:2]:raise ValueError('Target position grid differs')
            if source_blocks.shape[-1]!=self.block_size:raise ValueError('Trained block size differs')
            # input_ids contains only source-vocabulary clean anchors indexed
            # on the TARGET position grid; it is never a target-token label.
            anchors,keep,hidden=self._forward_draft_blocks(input_ids=input_ids,
                hidden_states=hidden_states,loss_mask=loss_mask)
            labels,weights=gather_source_blocks(source_blocks,source_validity,anchors,keep)
            hidden=hidden.reshape(labels.shape[0],labels.shape[1],self.block_size,-1)
            predecessor=torch.cat([labels[:,:,:1],labels[:,:,:-1]],dim=-1)
            def terms(h,y,w,p):
                t=self._dflash_objective_chunk_terms(h,y,w,p)
                return t.ce_loss_num,t.loss_den,t.correct_num,t.accuracy_den
            totals=None
            for start in range(0,labels.shape[1],self.objective_chunk_blocks):
                end=start+self.objective_chunk_blocks
                values=checkpoint(terms,hidden[:,start:end],labels[:,start:end],
                    weights[:,start:end],predecessor[:,start:end],use_reentrant=False)
                totals=values if totals is None else tuple(a+b for a,b in zip(totals,values))
            numerator,denominator,correct,count=totals
            if not bool(denominator>0):raise ValueError('No valid source supervision')
            return numerator/denominator,correct/count.clamp_min(1),{
                'sampled_blocks':keep.sum().detach(),'valid_source_labels':count.detach()}
    return CrossOnline
