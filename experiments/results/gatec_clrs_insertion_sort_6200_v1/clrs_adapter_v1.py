"""Serialized official task adapter; unchanged AGT token/commit substrate."""
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.agt import AGTv3

PROGRAM = tuple(range(8))
SPECIES = ('SWA','SWA','GQA','GQA','GDN','GDN','FFN','FFN')

def fields(value, prefix=''):
    if torch.is_tensor(value) or value is None or isinstance(value,(str,int,float,bool)):
        return {prefix:value}
    result={}
    for key,item in (value.items() if isinstance(value,dict) else vars(value).items()):
        result.update(fields(item,prefix+'/'+str(key)))
    return result

def snapshot(value):
    return {k:v.detach().clone() if torch.is_tensor(v) else v for k,v in fields(value).items()}

def assert_equal(a,b,exact=False):
    assert a.keys()==b.keys(), 'State schema mismatch'
    for k,x in a.items():
        y=b[k]
        if torch.is_tensor(x):
            assert torch.is_tensor(y) and x.dtype==y.dtype and x.shape==y.shape,k
            assert (torch.equal(x,y) if exact else torch.allclose(x,y,atol=1e-6,rtol=1e-5)),k
        else:assert type(x) is type(y) and x==y,k

def assert_disjoint(a,b):
    af,bf=fields(a),fields(b)
    assert af.keys()==bf.keys()
    for k,v in af.items():
        if torch.is_tensor(v) and v.numel():
            assert v.untyped_storage().data_ptr()!=bf[k].untyped_storage().data_ptr(),k

def metadata(value):
    return {k:dict(shape=list(v.shape),dtype=str(v.dtype)) if torch.is_tensor(v) else v
            for k,v in fields(value).items()}

def cpu_snapshot(value):
    return {k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in fields(value).items()}

class InsertionSortAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(6200)
        self.body=AGTv3(dim=64,num_memory_slots=16,vocab_size=2)
        self.body.executor.max_execution_depth=8
        assert tuple(b.species for b in self.body.blocks.blocks)==SPECIES
        for name,p in self.body.named_parameters():
            p.requires_grad_(not name.startswith(('controller.','output_head.')))
        self.encoder=nn.Linear(4,64)
        self.query=nn.Linear(64,64,bias=False)
        self.key=nn.Linear(64,64,bias=False)
        self.eval()

    def encode_features(self,keys,positions):
        n=len(keys)
        assert n==len(positions) and n>=2 and 2*n<=256
        x=torch.zeros(1,2*n,4,device=self.encoder.weight.device,dtype=torch.float32)
        x[0,:n,0]=1; x[0,n:,1]=1
        pos=torch.tensor(positions,device=x.device,dtype=x.dtype)
        x[0,:n,2]=pos; x[0,n:,2]=pos
        x[0,:n,3]=torch.tensor(keys,device=x.device,dtype=x.dtype)
        assert x[0,n:,3].eq(0).all()
        return x

    def forward(self,keys,positions,ledger,case_id,scripted=False,retain_gradients=False,retain_states=False):
        # No labels/hints are accepted by this computational interface.
        features=self.encode_features(keys,positions)
        x=self.encoder(features);n=len(keys)
        state=self.body.initial_persistent_state(1,x.device,x.dtype)
        candidates=[];logits=[];retained=[];candidate_keys=None;prefix_state=None
        for token in range(2*n):
            out=ledger.token(self.body,x[:,token],state,case_id,token,scripted)
            state=out['state']
            if retain_states:retained.append(cpu_snapshot(state))
            if token<n:
                candidate=state.s
                if retain_gradients and candidate.requires_grad:candidate.retain_grad()
                candidates.append(candidate)
                if token==n-1:
                    candidate_keys=self.key(torch.stack(candidates,dim=1))
                    prefix_state=state
                    if retain_gradients:
                        for value in fields(prefix_state).values():
                            if torch.is_tensor(value) and value.requires_grad:value.retain_grad()
            else:
                q=self.query(state.s).unsqueeze(1)
                scores=(q@candidate_keys.transpose(1,2)).squeeze(1)/(64**.5)
                assert scores.shape==(1,n) and bool(torch.isfinite(scores).all())
                logits.append(scores)
        assert state.tokens_committed==2*n
        return dict(pointer_logits=torch.stack(logits,dim=1),state=state,
                    candidates=candidates,prefix_state=prefix_state,
                    retained_states=retained,features=features)

def pointer_loss(logits,targets):
    n=logits.shape[-1]
    assert logits.shape==(1,n,n)
    labels=torch.tensor(targets,device=logits.device,dtype=torch.long)
    assert labels.shape==(n,) and bool(((labels>=0)&(labels<n)).all())
    return torch.nn.functional.cross_entropy(logits[0],labels)

def pointer_predictions(logits):
    return logits.detach().argmax(-1).cpu().tolist()
