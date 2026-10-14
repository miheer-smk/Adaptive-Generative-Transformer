"""Zero AGT-token unit checks; no model constructed."""
import ast,json,math
from pathlib import Path
import torch
from clrs_adapter_v1 import pointer_loss, fields, snapshot, assert_equal, assert_disjoint
from src.masks import build_execution_mask

OUT=Path(__file__).resolve().parent

def main():
    names=[]
    for n in (2,4,8,16,64):
        logits=torch.zeros(1,n,n,requires_grad=True)
        labels=list(range(n))
        # Loss arithmetic only, not a model forward/backward.
        assert abs(float(pointer_loss(logits,labels).detach())-math.log(n))<1e-6
    names.append('mean_pointer_CE_shapes_and_arithmetic')
    x=torch.ones(1,4);a={'s':x,'ctx':{'version':0}};b={'s':x.clone(),'ctx':{'version':0}}
    assert_equal(snapshot(a),snapshot(b),True);assert_disjoint(a,b)
    for bad in ({'s':x.clone(),'ctx':{'version':1}}, {'s':x.clone()}):
        try:assert_equal(snapshot(a),snapshot(bad),True)
        except AssertionError:pass
        else:raise AssertionError('Negative state comparison must fail')
    names.append('strict_metadata_and_schema_negatives')
    try:assert_disjoint(a,{'s':x,'ctx':{'version':0}})
    except AssertionError:pass
    else:raise AssertionError('Aliased tensor must fail')
    names.append('ownership_negative')
    for action in range(8):
        mask=build_execution_mask(max(0,action-1),action,False,8)
        assert mask.dtype==torch.bool and mask.shape==(11,) and bool(mask[action])
    names.append('canonical_script_legality')
    node=ast.parse((OUT/'clrs_adapter_v1.py').read_text())
    args=next(n for n in ast.walk(node) if isinstance(n,ast.FunctionDef) and n.name=='forward').args
    assert not any(x.arg in ('labels','pred','targets','hints') for x in args.args)
    names.append('computational_interface_excludes_labels_and_hints')
    seal=json.loads((OUT/'scope_lock.json').read_text())
    assert sum(x['tokens'] for x in seal['smoke_menu'])==512
    assert seal['backward_cap']==6 and seal['optimizer_updates']==0
    names.append('locked_menu_arithmetic')
    (OUT/'pure_unit_results_v1.json').write_text(json.dumps(dict(passed=len(names),failed=0,names=names,
        AGT_models_constructed=0,model_tokens=0,backwards=0,optimizer_objects=0),indent=2)+'\n')
    print('PASS',len(names),'pure unit checks; zero model calls')

if __name__=='__main__':main()
