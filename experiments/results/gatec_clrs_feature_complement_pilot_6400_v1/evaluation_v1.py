"""Prescribed native evaluations and accompanying captures, no extra trajectories."""
import json,time,math
import torch
from support_v1 import OUT,save
from observer_v1 import pointer_loss,NumericalFailure
from mechanistic_v1 import Capture
from metrics_v1 import metrics,aggregate

def score_record(row,out,loss,capture):
    n=row['n'];S=out['pointer_logits'][0].detach().cpu();y=torch.tensor(row['pred'],dtype=torch.long)
    Q=capture['projected_queries'].double();K=capture['projected_keys'].double()
    double=Q@K.T/8
    assert torch.allclose(S.double(),double,atol=1e-6,rtol=1e-5),'Native pointer score/projection mismatch'
    Qc=Q-Q.mean(0,keepdim=True);Kc=K-K.mean(0,keepdim=True);b=Kc@Q.mean(0)/8;I=Qc@Kc.T/8
    rowcenter=double-double.mean(1,keepdim=True);err=float((rowcenter-b[None,:]-I).abs().max())
    assert err<=1024*torch.finfo(torch.float64).eps*(1+float(double.abs().max()))
    common=b[None,:].expand(n,-1).float().clone()
    conditions=dict(native=metrics(S,y),query_independent=metrics(common,y),uniform=metrics(torch.zeros_like(S),y),self_pointer=metrics(None,y,list(range(n))),original_order=metrics(None,y,[0]+list(range(n-1))))
    pos=neg=cp=ap=fg=fb=0;sum_im=0.
    for i,t in enumerate(y.tolist()):
        for j in range(n):
            if j==t:continue
            cm=float(b[t]-b[j]);im=float(I[i,t]-I[i,j]);am=float(S[i,t]-S[i,j])
            pos+=im>0;neg+=im<0;cp+=cm>0;ap+=am>0;fg+=cm<=0 and am>0;fb+=cm>0 and am<=0;sum_im+=im
    detail=dict(identity_error=err,pair_comparisons=n*(n-1),interaction_improves=pos,interaction_worsens=neg,common_positive=cp,actual_positive=ap,flips_toward_target=fg,flips_against_target=fb,mean_interaction_target_margin=sum_im/(n*(n-1)),common_component_norm=float((b[None,:].expand(n,-1)).norm()),interaction_norm=float(I.norm()),mean_target_rank_native=conditions['native']['mean_target_rank'],mean_target_rank_common=conditions['query_independent']['mean_target_rank'])
    capture.update(scores=S,common_scores=common,common_candidate_preference=b,query_dependent_interaction=I)
    return dict(id=row['id'],n=n,original_input_sha256=row['original_input_sha256'],targets=row['pred'],conditions=conditions,native_loss_CE_FP32=float(loss.detach()),scores=S.tolist(),interaction=detail,uniform_analytic=dict(predecessor_accuracy=1/n,exact_array_accuracy=n**(-n),CE=math.log(n),note='Analytic independent random-pointer expectation only; uniform condition records deterministic lowest-index tie-breaking'))

def evaluate(model,rows,ledger,phase):
    ledger.phase=phase;records=[];captures=[];summaries=[]
    with (OUT/(ledger.arm+'_'+phase+'_records_v1.jsonl')).open('x') as f,(OUT/(ledger.arm+'_'+phase+'_mechanistic_v1.jsonl')).open('x') as m:
        for row in rows:
            ledger.check();ledger.array_attempted+=1;cap=Capture(model,ledger,row['n']);start=time.monotonic()
            try:
                with torch.no_grad():
                    out=model(row['key_fp32'],row['pos'],ledger,row['id'])
                    loss=pointer_loss(out['pointer_logits'],row['pred'])
                torch.cuda.synchronize()
                if not bool(torch.isfinite(loss)):raise NumericalFailure('Nonfinite evaluation CE')
                cap.close();data,summary=cap.finish(out,row);record=score_record(row,out,loss,data)
                record['elapsed_seconds']=time.monotonic()-start;record['commits']=out['state'].tokens_committed
                assert record['commits']==2*row['n']
                f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
                m.write(json.dumps(summary,allow_nan=False)+'\n');m.flush()
                records.append(record);captures.append(data);summaries.append(summary);ledger.array_completed+=1;ledger.persist()
                if len(records)%8==0:print('EVAL',ledger.arm,phase,len(records),'/',len(rows),'CE',record['native_loss_CE_FP32'],flush=True)
                del out,loss;ledger.check()
            except BaseException:
                partial=dict(completed=captures,active_case=row['id'],active_mixers=cap.mix,active_blocks=cap.blocks,active_pre_terminal_norm=cap.raw,active_committed=cap.committed)
                with (OUT/(ledger.arm+'_'+phase+'_partial_capture_v1.pt')).open('xb') as partial_file:torch.save(partial,partial_file)
                ledger.array_failed+=1;ledger.persist();raise
            finally:cap.close()
    with (OUT/(ledger.arm+'_'+phase+'_captures_v1.pt')).open('xb') as f:torch.save(captures,f)
    summary=aggregate(records);save(ledger.arm+'_'+phase+'_summary_v1.json',summary)
    return records,summary,summaries
