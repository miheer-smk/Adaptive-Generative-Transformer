"""Read-only captures during prescribed evaluation, plus CPU tensor summaries."""
import torch
from pilot_modules_v1 import active_mixers

def cloud(x):
    x=x.double();n=len(x);r=x.norm(dim=1);u=x/r.clamp_min(torch.finfo(torch.float64).tiny)[:,None]
    ij=torch.triu_indices(n,n,1);i,j=ij[0],ij[1];diff=x[i]-x[j];d2=diff.square().sum(1)
    radial=(r[i]-r[j]).square();angular=r[i]*r[j]*(u[i]-u[j]).square().sum(1)
    error=float((d2-radial-angular).abs().max()) if len(d2) else 0.
    assert error<=1e-10*(1+float(d2.max()) if len(d2) else 1)
    nonzero=(r[i]>0)&(r[j]>0);cos=(u[i]*u[j]).sum(1)[nonzero]
    centered=x-x.mean(0,keepdim=True)
    return dict(positions=n,finite=bool(torch.isfinite(x).all()),mean_norm=float(r.mean()),min_norm=float(r.min()),max_norm=float(r.max()),common_component_norm=float(x.mean(0).norm()),centered_energy=float(centered.square().sum()),mean_pair_distance=float(d2.sqrt().mean()),mean_pair_cosine=float(cos.mean()) if len(cos) else None,mean_angular_dispersion=float((1-cos).mean()) if len(cos) else None,mean_radial_squared=float(radial.mean()),mean_angular_squared=float(angular.mean()),radial_fraction=float(radial.sum()/d2.sum()) if d2.sum()>0 else None,radial_angular_identity_error=error)

def transport(x,y):
    x=x.double();y=y.double();ij=torch.triu_indices(len(x),len(x),1);dx=x[ij[0]]-x[ij[1]];dy=y[ij[0]]-y[ij[1]]
    nx=dx.square().sum(1);ny=dy.square().sum(1);keep=nx>0;ratio=(ny[keep]/nx[keep]).sqrt()
    return dict(input=cloud(x),output=cloud(y),observed_pairs=len(nx),nonzero_input_pairs=int(keep.sum()),mean_input_difference_squared=float(nx.mean()),mean_transported_difference_squared=float(ny.mean()),mean_observed_amplification=float(ratio.mean()) if len(ratio) else None,min_observed_amplification=float(ratio.min()) if len(ratio) else None,max_observed_amplification=float(ratio.max()) if len(ratio) else None,zero_input_pairs_output_squared_max=float(ny[~keep].max()) if (~keep).any() else None,interpretation='Actual corresponding position differences; not a worst-case singular-value inference')

class Capture:
    def __init__(self,model,ledger,n):
        self.n=n;self.ledger=ledger;self.handles=[];self.query=[];self.key=[];self.raw=[];self.committed=[]
        self.mix={str(i):dict(inputs=[],outputs=[]) for i in range(8)};self.blocks={str(i):dict(inputs=[],outputs=[]) for i in range(8)}
        def cpu(x):return x.detach().cpu().clone()
        def mhook(i):
            def hook(module,args,value):
                assert self.ledger.current_token==len(self.mix[str(i)]['inputs'])
                self.mix[str(i)]['inputs'].append(cpu(args[0]).reshape(1,64));self.mix[str(i)]['outputs'].append(cpu(value).reshape(1,64))
            return hook
        def bhook(i):
            def hook(module,args,value):
                value=value[0] if isinstance(value,tuple) else value
                self.blocks[str(i)]['inputs'].append(cpu(args[0]).reshape(1,64));self.blocks[str(i)]['outputs'].append(cpu(value).reshape(1,64))
            return hook
        def normhook(module,args,value):
            self.raw.append(cpu(args[0]).reshape(1,64));self.committed.append(cpu(value).reshape(1,64))
        for i,(name,module) in enumerate(active_mixers(model).items()):
            self.handles.append(module.register_forward_hook(mhook(i)))
            self.handles.append(model.body.blocks.blocks[i].register_forward_hook(bhook(i)))
        self.handles.append(model.body.commit.state_norm.register_forward_hook(normhook))
        self.handles.append(model.query.register_forward_hook(lambda m,a,o:self.query.append(cpu(o).reshape(1,64))))
        self.handles.append(model.key.register_forward_hook(lambda m,a,o:self.key.append(cpu(o).reshape(n,64))))
    def close(self):
        for h in self.handles:h.remove()
    def finish(self,output,row):
        assert len(self.query)==self.n and len(self.key)==1 and len(self.raw)==len(self.committed)==2*self.n
        data=dict(id=row['id'],input_sha256=row['original_input_sha256'],targets=torch.tensor(row['pred']),projected_queries=torch.cat(self.query),projected_keys=self.key[0],pre_terminal_norm=torch.cat(self.raw),committed=torch.cat(self.committed),mixers={},blocks={})
        assert torch.equal(data['committed'][:self.n],torch.cat([v.detach().cpu() for v in output['candidates']]))
        for family,source in [('mixers',self.mix),('blocks',self.blocks)]:
            for i,values in source.items():
                assert all(len(v)==2*self.n for v in values.values())
                data[family][i]={k:torch.cat(v) for k,v in values.items()}
        for i in range(8):
            assert torch.equal(data['blocks'][str(i)]['outputs'],data['blocks'][str(i)]['inputs']+data['mixers'][str(i)]['outputs'])
        assert all(torch.isfinite(t).all() for t in [data['projected_queries'],data['projected_keys'],data['pre_terminal_norm'],data['committed']])
        summary=dict(id=row['id'],n=self.n,roles={})
        for role,sl in [('candidates',slice(0,self.n)),('queries',slice(self.n,2*self.n))]:
            self.ledger.check()
            summary['roles'][role]=dict(mixers={i:transport(v['inputs'][sl],v['outputs'][sl]) for i,v in data['mixers'].items()},blocks={i:transport(v['inputs'][sl],v['outputs'][sl]) for i,v in data['blocks'].items()},terminal_normalization=transport(data['pre_terminal_norm'][sl],data['committed'][sl]))
        return data,summary
