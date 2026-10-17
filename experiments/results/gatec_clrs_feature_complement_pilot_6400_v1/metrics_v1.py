"""Unchanged established score/control metrics, without module-level grad changes."""
import torch
def invalid(p):
    n=len(p)
    if any(not isinstance(v,int) or not 0<=v<n for v in p):return True
    roots=[i for i,v in enumerate(p) if i==v]
    if len(roots)!=1:return True
    successors=[[] for _ in range(n)]
    for i,v in enumerate(p):
        if i!=v:successors[v].append(i)
    if any(len(v)>1 for v in successors):return True
    seen=set();node=roots[0]
    while node not in seen:
        seen.add(node)
        if not successors[node]:return len(seen)!=n
        node=successors[node][0]
    return True

def metrics(scores,target,pred=None):
    n=len(target);y=target.tolist()
    if scores is None:
        assert pred is not None
        return dict(predictions=pred,predecessor_accuracy=sum(a==b for a,b in zip(pred,y))/n,exact_array_accuracy=int(pred==y),invalid_chain=int(invalid(pred)),
            CE=None,target_ranks=None,stable_index_ranks=None,target_margins=None,distinct_predicted_candidates=len(set(pred)),score_metrics='Unavailable for hard-only control; no probability smoothing')
    S=scores.cpu().double();assert S.shape==(n,n) and torch.isfinite(S).all()
    p=S.argmax(1).tolist();shift=S-S.max(1,keepdim=True).values;logp=shift-torch.logsumexp(shift,1,keepdim=True)
    ce=-logp[torch.arange(n),target].mean();ranks=[];stable=[];margins=[];ties=[]
    for i,t in enumerate(y):
        ranks.append(1+int((S[i]>S[i,t]).sum()));ties.append(int((S[i]==S[i,t]).sum()))
        stable.append(ranks[-1]+int(((S[i]==S[i,t])&(torch.arange(n)<t)).sum()))
        wrong=S[i].clone();wrong[t]=-torch.inf;margins.append(float(S[i,t]-wrong.max()))
    return dict(predictions=p,predecessor_accuracy=sum(a==b for a,b in zip(p,y))/n,exact_array_accuracy=int(p==y),invalid_chain=int(invalid(p)),CE=float(ce),
        target_ranks=ranks,stable_index_ranks=stable,target_ties=ties,target_margins=margins,mean_target_rank=sum(ranks)/n,
        mean_stable_index_rank=sum(stable)/n,mean_target_margin=sum(margins)/n,distinct_predicted_candidates=len(set(p)),mean_max_probability=float(logp.exp().max(1).values.mean()))

def aggregate(rows):
    names=rows[0]['conditions'].keys();out={}
    for name in names:
        values=[x['conditions'][name] for x in rows]
        keys=['predecessor_accuracy','exact_array_accuracy','invalid_chain','distinct_predicted_candidates','CE','mean_target_rank','mean_stable_index_rank','mean_target_margin','mean_max_probability']
        out[name]={k:sum(v[k] for v in values)/len(values) if all(v.get(k) is not None for v in values) else None for k in keys}
        out[name]['arrays']=len(values)
    return out

