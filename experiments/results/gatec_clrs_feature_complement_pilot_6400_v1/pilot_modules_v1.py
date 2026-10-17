"""The sole scientific amendment: exact feature complement at eight bank mixers."""
import types
import torch

ACTIVE_NAMES=tuple('body.blocks.blocks.'+str(i)+('.mhc' if i<6 else '.core.mhc') for i in range(8))
EXCLUDED_NAME='body.prelude_block.mhc'
def feature_complement_forward(self,x):
    assert self.dim==64 and x.shape[-1]==64
    matrix=self.mixing_matrix()
    Q=torch.eye(64,device=x.device,dtype=x.dtype)-torch.ones((64,64),device=x.device,dtype=x.dtype)/64
    return torch.matmul(x,(matrix+Q).T)*self.scale

def active_mixers(model):
    modules=dict(model.named_modules())
    assert tuple(b.species for b in model.body.blocks.blocks)==('SWA','SWA','GQA','GQA','GDN','GDN','FFN','FFN')
    assert all(n in modules and modules[n].dim==64 for n in ACTIVE_NAMES)
    assert modules[EXCLUDED_NAME].dim==64
    return {n:modules[n] for n in ACTIVE_NAMES}

def install_treatment(model):
    targets=active_mixers(model)
    for module in targets.values():
        assert 'forward' not in module.__dict__, 'No duplicate amendment'
        module.forward=types.MethodType(feature_complement_forward,module)
    assert 'forward' not in dict(model.named_modules())[EXCLUDED_NAME].__dict__
    return list(targets)
