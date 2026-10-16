"""CPU-only algebra on captured tensors. No AGT imports or model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,csv,math,hashlib,importlib.util
from pathlib import Path
sys.path.append('/tmp/agt_clrs_cpu_6200_v1/lib/python3.12/site-packages')
import torch
torch.set_grad_enabled(False);torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent
PRIOR=OUT.parent/'gatec_clrs_residual_score_audit_6200_v1'
spec=importlib.util.spec_from_file_location('prior_offline_numeric',PRIOR/'audit_numeric_v3.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)

def save(name,data):
    with (OUT/name).open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def csvsave(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def energy(x):return float(x.square().sum())
def center(x):return x-x.mean(0,keepdim=True)
def ratio(a,b):return a/b if b else None

def main():
    result=json.loads((OUT/'capture_results_v1.json').read_text());assert result['status']=='COMPLETED_ONE_FROZEN_FORWARD'
    data=torch.load(OUT/'captured_tensors_v1.pt',map_location='cpu',weights_only=True)
    derived=torch.load(PRIOR/'derived_algebra_v1.pt',map_location='cpu',weights_only=True)['final']
    oldres=json.loads((PRIOR/'residual_energy_v1.json').read_text())
    metrics=[];pairs=[];summary=[];recon=[]
    for i in range(8):
        name='body.blocks.blocks.'+str(i)+('.core.mhc' if i>=6 else '.mhc')
        A=derived['mixer/'+name]['A'];U,s,Vh=prior.svd_cpu(A);right=Vh[0];left=U[:,0]
        assert A.shape==(64,64)
        for group,sl in [('candidate',slice(0,16)),('query',slice(16,32))]:
            X=data['mhc'][str(i)]['inputs'][sl].double();Y=data['mhc'][str(i)]['outputs'][sl].double()
            H=data['blocks'][str(i)]['inputs'][sl].double();B=data['blocks'][str(i)]['outputs'][sl].double()
            expected=next(x for x in oldres if x['model']=='final' and x['positions']==group and x['block']==i)
            stats={};radials={}
            for stage,x in [('mixer_input',X),('mixer_output',Y),('block_input',H),('block_output',B)]:
                stats[stage]=prior.stats(x);radialrows=[]
                radials[stage]=prior.radial(x,'final',group,stage,radialrows)
                norms=x.norm(dim=1)
                for v in radialrows:
                    a,b=v['i'],v['j']
                    pairs.append(dict(block=i,positions=group,stage=stage,i=a,j=b,
                        squared_distance=v['total_squared_distance'],distance=math.sqrt(v['total_squared_distance']),
                        cosine=float(x[a]@x[b]/(norms[a]*norms[b])) if norms[a]*norms[b]>0 else None,
                        radial_squared=v['radial_squared'],angular_squared=v['angular_squared'],identity_error=v['identity_error']))
            xc,yc,hc,bc=map(center,[X,Y,H,B]);ex,ey,eh,eb=map(energy,[xc,yc,hc,bc])
            projected=X@A.T
            reconstruction=dict(block=i,positions=group,max_absolute_error=float((Y-projected).abs().max()),
                approved_tolerance_pass=bool(torch.allclose(Y,projected,atol=1e-6,rtol=1e-5)),
                provenance='Actual CUDA input/output compared with source-equation CPU-FP32 A retained by prior audit; comparison is parameter algebra, not another model call')
            recon.append(reconstruction)
            # FP32 residual additions were checked bitwise during capture. Float64
            # difference is recorded, not mistaken for an extra hidden operation.
            addition_error=float((B-H-Y).abs().max())
            cross=2*float((hc*yc).sum());increment_from_block=B-H
            assert math.isclose(eh,expected['centered_input_energy'],rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(eb,expected['centered_output_energy'],rel_tol=1e-12,abs_tol=1e-12)
            def alignment(x,v):
                mu=x.mean(0);c=center(x)
                return dict(common_abs_cosine=abs(float(mu@v))/float(mu.norm()) if mu.norm()>0 else None,
                    centered_energy_fraction=ratio(energy((c@v).reshape(-1,1)),energy(c)),
                    all_energy_fraction=ratio(energy((x@v).reshape(-1,1)),energy(x)))
            # Rotation here means geometric coordinate-direction change, not an
            # orthogonal operator or a claim of preserved information.
            normsX=X.norm(dim=1);normsY=Y.norm(dim=1)
            cosXY=[float(X[j]@Y[j]/(normsX[j]*normsY[j])) if normsX[j]*normsY[j]>0 else None for j in range(16)]
            pnormX=xc.norm(dim=1);pnormY=yc.norm(dim=1)
            cosC=[float(xc[j]@yc[j]/(pnormX[j]*pnormY[j])) if pnormX[j]*pnormY[j]>0 else None for j in range(16)]
            metrics.append(dict(block=i,species=('SWA','SWA','GQA','GQA','GDN','GDN','FFN','FFN')[i],positions=group,module=name,
                stage_statistics=stats,radial_statistics=radials,
                centered_energy_ratio=ratio(ey,ex),centered_energy_change=ey-ex,
                common_energy_ratio=ratio(stats['mixer_output']['common_energy'],stats['mixer_input']['common_energy']),
                input_centered_energy_fraction=ratio(ex,stats['mixer_input']['total_energy']),
                output_centered_energy_fraction=ratio(ey,stats['mixer_output']['total_energy']),
                input_alignment_to_dominant_right=alignment(X,right),output_alignment_to_dominant_left=alignment(Y,left),
                input_alignment_to_uniform=alignment(X,torch.ones(64,dtype=torch.float64)/8),
                same_position_input_output_cosines=cosXY,centered_same_position_cosines=cosC,
                residual=dict(input_centered_energy=eh,output_centered_energy=eb,twice_centered_cross=cross,
                    mixer_output_centered_energy=ey,float64_residual_coordinate_error=addition_error,
                    fp32_addition_bitwise_verified=True,retained_prior_block_input_energy=expected['centered_input_energy'],
                    retained_prior_block_output_energy=expected['centered_output_energy'],
                    prior_centered_increment_energy=expected['centered_increment_energy'],
                    actual_delta_minus_mixer_max_error=float((increment_from_block-Y).abs().max()),
                    note='Actual mixer output precedes FP32 residual addition; subtracting two stored block states can lose tiny increment coordinates through rounding.'),
                effective_operator_reconstruction=reconstruction,
                task_relevance='No block-input task-readout or intervention; positional distinctions alone are not established as task-useful'))
            summary.append(dict(block=i,species=('SWA','SWA','GQA','GQA','GDN','GDN','FFN','FFN')[i],positions=group,
                input_centered_energy=ex,output_centered_energy=ey,energy_ratio=ratio(ey,ex),
                input_mean_pair_cosine=stats['mixer_input']['pairwise_cosine_mean'],output_mean_pair_cosine=stats['mixer_output']['pairwise_cosine_mean'],
                input_radial_fraction=radials['mixer_input']['radial_fraction'],output_radial_fraction=radials['mixer_output']['radial_fraction'],
                input_mean_pair_distance=stats['mixer_input']['mean_pairwise_L2'],output_mean_pair_distance=stats['mixer_output']['mean_pairwise_L2'],
                block_input_centered_energy=eh,block_output_centered_energy=eb,twice_residual_cross=cross,
                centered_increment_fraction_leading_output=alignment(Y,left)['centered_energy_fraction']))
    save('blockwise_metrics_v1.json',metrics);csvsave('blockwise_summary_v1.csv',summary);csvsave('pairwise_geometry_v1.csv',pairs)
    save('operator_reconstruction_checks_v1.json',recon)
    # Original scores/state replay establishes unchanged computation. These
    # algebraic checks supplement it without acting as another model test.
    save('analysis_verification_v1.json',dict(success=True,block_group_comparisons=len(metrics),pair_stage_comparisons=len(pairs),
        all_prior_block_energies_reused=True,source_equation_operator_matches=all(r['approved_tolerance_pass'] for r in recon),
        max_operator_absolute_error=max(r['max_absolute_error'] for r in recon),offline_only=True,additional_model_calls=0,
        additional_backwards=0,no_inverse=True,SVD='installed CPU SciPy gesvd, unchanged arrays'))
    save('evidence_limits_v1.json',dict(direct=['256 actual mHC inputs and outputs, detached after invocation','Corresponding raw residual inputs and block outputs','One complete canonical trajectory and committed-state summaries','Exact retained-output replay'],
        algebraic=['Centered energies, singular spectra, radial/angular decomposition, pair geometry','Dominant directions from previously retained effective A','CPU algebraic A replay; no mixer/module execution'],
        not_captured=['Intermediates inside attention/GDN/FFN cores before their output projections','Full information content/readability of all persistent-state channels','Historical training trajectory of these tensors','Counterfactual identity-mixer or architecture-change outcomes','Other arrays or seeds'],
        interpretation='Compression of positional distinctions is directly observed where present. Their task usefulness and causal contribution to historical failure remain unestablished. Residual addition and normal contextual commits remain active.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
