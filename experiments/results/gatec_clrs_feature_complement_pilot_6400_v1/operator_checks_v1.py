"""Only isolated MHC fixtures; no AGT construction, trajectory or backward."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,types
from support_v1 import ROOT,OUT,save,integrity
sys.path.insert(0,str(ROOT))
import torch
from src.mhc import MHC
from pilot_modules_v1 import feature_complement_forward,ACTIVE_NAMES
torch.set_num_threads(1)
def main():
    assert not (OUT/'operator_checks_v1.json').exists(),'No check rerun'
    assert integrity()['success']
    assert torch.cuda.is_available(),'Required DGX CUDA unavailable'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[]
    for device,dtype in [('cpu',torch.float32),('cpu',torch.float64),('cuda',torch.float32)]:
        control=MHC(dim=64).to(device=device,dtype=dtype);treat=MHC(dim=64).to(device=device,dtype=dtype)
        with torch.no_grad():
            z=torch.arange(4096,device=device,dtype=dtype).reshape(64,64)
            control.logits.copy_(torch.sin(z*.031)+torch.cos(z*.007))
            control.scale.copy_(torch.linspace(.5,1.5,64,device=device,dtype=dtype));treat.load_state_dict(control.state_dict())
            treat.forward=types.MethodType(feature_complement_forward,treat)
            P=control.mixing_matrix();Q=torch.eye(64,device=device,dtype=dtype)-torch.ones((64,64),device=device,dtype=dtype)/64;u=torch.ones(64,device=device,dtype=dtype)/8
            assert torch.allclose(Q@u,torch.zeros_like(u),atol=1e-6,rtol=1e-5)
            assert torch.allclose(Q@Q,Q,atol=1e-6,rtol=1e-5) and torch.equal(Q,Q.T)
            for shape in [(3,64),(2,3,64),(1,1,64)]:
                x=torch.sin(torch.arange(__import__('math').prod(shape),device=device,dtype=dtype).reshape(shape)*.017)
                saved=x.clone();state={k:v.clone() for k,v in treat.state_dict().items()}
                y=treat(x);expected=(x@(P+Q).T)*control.scale
                assert y.shape==x.shape and y.dtype==dtype and y.device==x.device and torch.equal(x,saved)
                assert torch.equal(y,expected)
                assert all(torch.equal(v,treat.state_dict()[k]) for k,v in state.items())
                assert torch.allclose(x@Q.T,x-x.mean(-1,keepdim=True),atol=1e-6,rtol=1e-5)
                assert torch.allclose(y,control(x)+(x@Q.T)*control.scale,atol=1e-6,rtol=1e-5)
                assert not torch.allclose(y,(x@(P+Q))*control.scale,atol=1e-6,rtol=1e-5),'Fixture must detect wrong transpose'
                assert not torch.allclose(y,control(x)+x@Q.T,atol=1e-6,rtol=1e-5),'Fixture must detect gain moved off Q'
                results.append(dict(device=device,dtype=str(dtype),shape=list(shape),exact_formula=True,Q_uniform_annihilation=True,complement_action=True,transpose_check=True,gain_placement=True,input_and_parameters_unchanged=True))
    torch.cuda.synchronize()
    save('operator_checks_v1.json',dict(success=True,fixtures=results,CPU_cases=6,CUDA_cases=3,skips=0,full_AGT_constructions=0,model_tokens=0,backwards=0,optimizer_objects=0,sites=list(ACTIVE_NAMES),torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(),interpretation='Synthetic operator fixtures only, not pilot task outcomes'))
    print(json.dumps(dict(operator_checks='PASS',CPU_cases=6,CUDA_cases=3,full_model_calls=0),indent=2))
if __name__=='__main__':main()
