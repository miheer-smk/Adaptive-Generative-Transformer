"""Offline family-clustered descriptive and preregistered qualification analysis."""
import random, math

def quantile(values,q):
    a=sorted(values); h=(len(a)-1)*q; i=int(h); f=h-i
    return a[i]*(1-f)+a[min(i+1,len(a)-1)]*f

def metric(rows):
    n=len(rows)
    return {'episodes':n,'complete_accuracy':sum(r['S'] for r in rows)/n,
            'query_accuracy':sum(r['query_accuracy'] for r in rows)/n,
            'mean_CE':sum(r['CE'] for r in rows)/n,
            'mean_margin':sum(r['margin'] for r in rows)/n}

def qualify(rows):
    programs=('B','A','R'); means={}; results={}; grouped={}
    for L in (64,96):
        part=[r for r in rows if r['length']==L]
        means[str(L)]={p:metric([r for r in part if r['program']==p]) for p in programs}
        ids=sorted({r['id'] for r in part}); families=sorted({r['family'] for r in part})
        table={eid:[next(r['S'] for r in part if r['id']==eid and r['program']==p) for p in programs] for eid in ids}
        oracle=sum(max(v) for v in table.values())/128
        best=max(v['complete_accuracy'] for v in means[str(L)].values())
        unique={p:sum(v[i]==1 and sum(v)==1 for v in table.values())/128 for i,p in enumerate(programs)}
        grouped[L]=[[table[eid] for eid in ids if next(r['family'] for r in part if r['id']==eid)==f] for f in families]
        assert len(grouped[L])==32 and all(len(f)==4 for f in grouped[L])
        results[str(L)]={'oracle':oracle,'strongest_observed_fixed':best,'G':oracle-best,'unique_success':unique}
    rng=random.Random(6103); bank={L:{p:[] for p in programs} for L in (64,96)}; gaps={64:[],96:[]}
    for _ in range(20000):
        for L in (64,96):
            counts=[0]*32
            for j in range(32): counts[rng.randrange(32)]+=1
            sums=[0,0,0]; oracle=0
            for f,count in zip(grouped[L],counts):
                if not count:continue
                for v in f:
                    oracle+=count*max(v)
                    for k in range(3):sums[k]+=count*v[k]
            for k,p in enumerate(programs):bank[L][p].append(sums[k]/128)
            gaps[L].append((oracle-max(sums))/128)
    for L in (64,96):
        for p in programs:means[str(L)][p]['complete_accuracy_interval95']=[quantile(bank[L][p],.025),quantile(bank[L][p],.975)]
        results[str(L)]['G_lower_adjusted']=quantile(gaps[L],.025)
    solvable=any(all(means[str(L)][p]['query_accuracy']>=.95 and means[str(L)][p]['complete_accuracy']>=.80 for L in (64,96)) for p in programs)
    headroom=all(results[str(L)]['strongest_observed_fixed']<.95 for L in (64,96))
    complementary=all(results[str(L)]['G']>=.10 and results[str(L)]['G_lower_adjusted']>0 and sum(x>=.05 for x in results[str(L)]['unique_success'].values())>=2 for L in (64,96))
    status='solvability failure' if not solvable else 'saturation failure' if not headroom else 'complementarity failure' if not complementary else 'CURRENT_GATES_PASS'
    return {'by_length_program':means,'complementarity':results,'solvability_pass':solvable,'headroom_pass':headroom,'complementarity_pass':complementary,'status':status,'bootstrap_seed':6103,'bootstrap_replicates':20000,'cluster':'four-episode family','interpretation':'conditional input-sampling uncertainty, not seed reproducibility or achievable selector performance'}

def ablation_summary(normal,ablated):
    pairs=[]; drop={}
    for p in ('B','A','R'):
        sub=[r for r in ablated if r['program']==p]
        assert len(sub)==16
        signed=[]
        for row in sub:
            match=next(r for r in normal if r['program']==p and r['id']==row['id'])
            delta=match['S']-row['S'];signed.append(delta)
            pairs.append({'id':row['id'],'family':row['family'],'program':p,'U':match['S'],'A':row['S'],'signed_decrease':delta})
        drop[p]=sum(signed)/16
    flag=drop['A']>=.05 and (drop['B']>=.05 or drop['R']>=.05)
    return {'descriptive_only':True,'independent_families':4,'confidence_intervals':None,'point_decreases':drop,'apparent_relevance_flag':flag,'pairs':pairs,'status':'provisionally qualified, inferential operation relevance unconfirmed' if flag else 'preliminary relevance unsupported'}
