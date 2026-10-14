"""Locked keyed-register task, deterministic generation and independent validation."""
from pathlib import Path
import hashlib, json, random, collections
import torch

OPS = ('SET', 'XOR', 'COPY', 'QUERY')

def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def below(rng, n):
    if n == 1:
        return 0
    while True:
        x = rng.getrandbits(n.bit_length())
        if x < n:
            return x

def permutation(rng, values):
    a = list(values)
    for i in range(len(a)-1, 0, -1):
        j = below(rng, i+1)
        a[i], a[j] = a[j], a[i]
    return a

def check_command(c):
    assert len(c) == 4
    op,k,j,b = c
    assert op in OPS and type(k) is int and 0 <= k < 8
    if op in ('SET','XOR'):
        assert j is None and type(b) is int and b in (0,1)
    elif op == 'COPY':
        assert type(j) is int and 0 <= j < 8 and b is None
    else:
        assert j is None and b is None

def interpret(commands):
    registers = [0]*8
    labels, positions, states = [], [], []
    for t,c in enumerate(commands):
        check_command(c)
        op,k,j,b = c
        if op == 'SET': registers[k] = b
        elif op == 'XOR': registers[k] ^= b
        elif op == 'COPY': registers[k] = int(registers[j])
        else:
            positions.append(t); labels.append(registers[k]); states.append(list(registers))
    return positions,labels,states

def independent_label(prefix):
    """Replay with a dictionary and separate dispatch; scalar COPY has no aliases."""
    state = dict.fromkeys(range(8),0)
    for op,dest,source,bit in prefix:
        if op == 'SET': state[dest] = int(bit)
        if op == 'XOR': state[dest] = (state[dest]+int(bit))%2
        if op == 'COPY':
            old = int(state[source]); state[dest] = old
    assert prefix[-1][0] == 'QUERY'
    return state[prefix[-1][1]]

def encode(commands):
    x = torch.zeros(len(commands),64,dtype=torch.float32)
    for t,c in enumerate(commands):
        check_command(c)
        op,k,j,b = c
        x[t,OPS.index(op)] = 1
        x[t,4+k] = 1
        if op == 'COPY': x[t,12+j] = 1
        if op in ('SET','XOR'): x[t,20+b] = 1
    return x.unsqueeze(0)

def candidate(rng,L):
    order = permutation(rng, range(8))
    bits = [rng.getrandbits(1) for _ in range(8)]
    a,d = order[:2]
    ordinary = [k for k in range(8) if k not in (a,d)]
    base = [['SET',k,None,bits[k]] for k in order]
    base += [['XOR',a,None,rng.getrandbits(1)], ['XOR',d,None,rng.getrandbits(1)]]
    qp = [8+(r+1)*((L-8)//8)-1 for r in range(8)]
    qkeys = permutation(rng,[k for k in range(8) if k != a])+[a]
    qmap = dict(zip(qp,qkeys))
    for t in range(10,L):
        if t in qmap:
            base.append(['QUERY',qmap[t],None,None]); continue
        op = OPS[below(rng,3)]
        if op == 'SET':
            k = ordinary[below(rng,6)]; base.append([op,k,None,rng.getrandbits(1)])
        elif op == 'XOR':
            k = below(rng,8); base.append([op,k,None,rng.getrandbits(1)])
        else:
            k = ordinary[below(rng,6)]; sources = [j for j in ordinary if j != k]
            base.append([op,k,sources[below(rng,5)],None])
    variants = []
    for v in range(4):
        c = [list(row) for row in base]
        if v >= 2:
            for row in c:
                if row[0] == 'SET': row[3] ^= 1
        if v%2:
            c[8][3] ^= 1; c[9][3] ^= 1
        variants.append(c)
    return variants,a,d

def validate_family(variants,a,d):
    L = len(variants[0]); assert L in (64,96)
    results = [interpret(c) for c in variants]
    for v,c in enumerate(variants):
        q,y,states = results[v]
        assert len(c) == L and q == [8+(r+1)*((L-8)//8)-1 for r in range(8)]
        assert {row[1] for row in c[:8]} == set(range(8)) and all(row[0]=='SET' for row in c[:8])
        assert all(independent_label(c[:t+1]) == label for t,label in zip(q,y))
        x = encode(c)
        assert x.dtype == torch.float32 and x.shape == (1,L,64)
        assert not x[:,:,22:].count_nonzero() and not x[:,q,12:22].count_nonzero()
        assert x[0,:,0:4].sum(1).eq(1).all() and x[0,:,4:12].sum(1).eq(1).all()
        for row in c[10:]:
            if row[0]=='SET': assert row[1] not in (a,d)
            if row[0]=='COPY': assert row[1] not in (a,d) and row[2] not in (a,d) and row[1]!=row[2]
    for left,right in ((0,1),(2,3)):
        c,e = variants[left],variants[right]
        assert c[10:] == e[10:] and c[-16:] == e[-16:]
        assert [(o,k,j) for o,k,j,b in c] == [(o,k,j) for o,k,j,b in e]
        assert [row for row in c if row[0]=='SET'] == [row for row in e if row[0]=='SET']
        for r,t in enumerate(results[left][0]):
            assert sum(results[left][2][r])%2 == sum(results[right][2][r])%2
            assert sum(row[3] for row in c[:t+1] if row[0] in ('SET','XOR'))%2 == sum(row[3] for row in e[:t+1] if row[0] in ('SET','XOR'))%2
        assert results[left][1][-1] != results[right][1][-1]
    for r in range(8): assert sum(result[1][r] for result in results)==2
    assert results[2][1] == [1-b for b in results[0][1]]
    assert results[3][1] == [1-b for b in results[1][1]]
    return results

def signatures(commands):
    q,_,_ = interpret(commands)
    return {'input':{digest(commands)},'initial':{digest(commands[:8])},
            'history':{digest(commands[:t]) for t in q},
            'template':{digest([[o,k,j] for o,k,j,b in commands])}}

def shortcut_predictions(commands):
    q,labels,states = interpret(commands)
    names = ('constant0','constant1','latest_SET','last_operand','global_XOR','inverse_global_XOR','window16')
    pred = {n:[] for n in names}
    for r,t in enumerate(q):
        k = commands[t][1]; prefix = commands[:t+1]
        latest = next(row[3] for row in reversed(prefix) if row[0]=='SET' and row[1]==k)
        operand = next(row[3] for row in reversed(prefix) if row[0] in ('SET','XOR'))
        window = independent_label(prefix[max(0,len(prefix)-16):])
        values = (0,1,latest,operand,sum(states[r])%2,1-sum(states[r])%2,window)
        for n,val in zip(names,values): pred[n].append(val)
    return {n: {'predictions':v,'S':int(v==labels),'query_accuracy':sum(a==b for a,b in zip(v,labels))/8} for n,v in pred.items()}

def seal_inputs(out, protected):
    out = Path(out)
    exclusions = {n:set() for n in ('input','initial','history','template')}
    exposure = []
    def find_commands(value):
        found=[]
        if isinstance(value,dict):
            if 'commands' in value:
                try:
                    commands=value['commands']; interpret(commands)
                    if commands: found.append(commands)
                except (AssertionError,TypeError,ValueError,KeyError): pass
            for v in value.values(): found.extend(find_commands(v))
        elif isinstance(value,list):
            for v in value: found.extend(find_commands(v))
        return found
    root = out.parents[2]
    for rel,identity in protected.items():
        if not rel.startswith('experiments/') or not rel.endswith('.json') or not any(s in rel.lower() for s in ('input','sequence','panel','manifest')): continue
        record={'path':rel,'sha256':identity.get('sha256')}
        p=root/rel
        try:
            comparable=find_commands(json.loads(p.read_text()))
            for c in comparable:
                for key,vals in signatures(c).items(): exclusions[key].update(vals)
            record.update(status='COMPARABLE' if comparable else 'NO_COMPARABLE_COMMAND_INPUTS',comparable_inputs=len(comparable))
        except (ValueError,UnicodeError,OSError) as e: record.update(status='UNKNOWN_UNREADABLE',error=type(e).__name__)
        exposure.append(record)
    panels={}; rejected=[]; encoded={}; manifest=[]
    for split,seed,nfamilies in (('training',6100,16),('development',6101,32)):
        rng=random.Random(seed); rows=[]; candidates=0
        for L in (64,96):
            accepted=0
            while accepted<nfamilies:
                candidates+=1
                if candidates>100000: raise RuntimeError('GENERATION_BLOCKED: candidate ceiling')
                variants,a,d=candidate(rng,L); results=validate_family(variants,a,d)
                sigs={key:set().union(*(signatures(c)[key] for c in variants)) for key in exclusions}
                conflicts=[key for key in exclusions if sigs[key]&exclusions[key]]
                if conflicts:
                    rejected.append(dict(split=split,L=L,candidate=candidates,family_sha256=digest(variants),reasons=conflicts));continue
                family=f'{split}/L{L}/family{accepted:03d}'
                for v,c in enumerate(variants):
                    q,y,_=results[v]; eid=f'{family}/V{v}'
                    row=dict(id=eid,family=family,variant=v,length=L,commands=c,query_positions=q,labels=y,active_key=a,decoy_key=d,input_sha256=digest(c),label_sha256=digest(y),family_sha256=digest(variants))
                    rows.append(row); encoded[eid]=encode(c)
                    manifest.append({k:val for k,val in row.items() if k!='commands'})
                for key,vals in sigs.items(): exclusions[key].update(vals)
                accepted+=1
        panels[split]=rows
    order=list(range(128));random.Random(6102).shuffle(order)
    diagnostic=[r['id'] for r in panels['development'] if int(r['family'].split('family')[-1])<2]
    assert len(diagnostic)==16 and len(panels['training'])==128 and len(panels['development'])==256
    shortcuts={}
    shortcut_rows={r['id']:shortcut_predictions(r['commands']) for r in panels['development']}
    for L in (64,96):
        members=[r for r in panels['development'] if r['length']==L]
        shortcuts[str(L)]={name:{'robust_accuracy':sum(shortcut_rows[r['id']][name]['S'] for r in members)/128,'query_accuracy':sum(shortcut_rows[r['id']][name]['query_accuracy'] for r in members)/128} for name in next(iter(shortcut_rows.values()))}
    passed=all(v['robust_accuracy']<.80 for row in shortcuts.values() for v in row.values())
    for name,value in [('training_inputs',panels['training']),('development_inputs',panels['development']),('input_manifest',manifest),('training_order',order),('diagnostic_selection',diagnostic),('exposure_inventory',exposure),('generation_rejections',rejected),('shortcut_records',shortcut_rows),('shortcut_screen',{'passes':passed,'by_length':shortcuts}),('symbolic_validation',{'passes':True,'episodes':384,'families':96,'independent_labels':3072,'counterfactual_properties_pass':True,'encoding_properties_pass':True,'split_history_disjoint':True,'reserved_unused_seeds':[6104,6105],'historical_exposure_limitations':'Only retained comparable command inputs excluded; binary parity inputs are a different alphabet; unretained exposure unknown'})]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    torch.save(encoded,out/'sealed_encoded_inputs.pt')
    return panels,encoded,order,diagnostic,passed
