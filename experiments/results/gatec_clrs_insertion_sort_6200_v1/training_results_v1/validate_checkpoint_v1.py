"""Read the new final checkpoint on CPU; no model or optimizer construction."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,collections
from pathlib import Path
import torch
from offline_support_v1 import OUT,ENGINEERING,sha,save

def main():
    identity=json.loads((OUT/'checkpoint_identity_v1.json').read_text())
    p=OUT/identity['path']
    assert sha(p)==identity['sha256'] and p.stat().st_size==identity['size']
    c=torch.load(p,map_location='cpu',weights_only=True)
    assert c['optimizer_update_count']==1000 and c['model_seed']==6200
    assert c['training_order_sha256']==sha(OUT/'training_order_v1.json')
    assert c['engineering_seal_sha256']==sha(ENGINEERING/'implementation_seal_v1.json')
    assert c['training_implementation_seal_sha256']==sha(OUT/'training_implementation_seal_v1.json')
    assert all(torch.isfinite(v).all() for v in c['model_state_dict'].values())
    opt=c['optimizer_state_dict']
    for g in opt['param_groups']:
        assert g['lr']==.001 and g['weight_decay']==0 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8
    steps=collections.Counter()
    for state in opt['state'].values():
        for v in state.values():
            if torch.is_tensor(v):assert torch.isfinite(v).all()
        steps[int(state['step'])]+=1
    save('checkpoint_validation_v1.json',dict(success=True,sha256=sha(p),optimizer_update_count=1000,
        model_state_tensors=len(c['model_state_dict']),optimizer_state_entries=len(opt['state']),
        parameter_state_step_histogram=dict(steps),all_tensors_finite=True,metadata_and_seals_match=True,
        model_constructions=0,optimizer_constructions=0,model_token_calls=0,backwards=0,optimizer_steps=0))
    print('PASS: final checkpoint identity, metadata, finite tensors and optimizer state')

if __name__=='__main__':main()
