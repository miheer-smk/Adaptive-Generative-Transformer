"""Prospectively sealed content-only panels, encoding and paired update orders."""
import ctypes
import hashlib
import json
from pathlib import Path
import random
import torch

BIT_KEYS = {'bits','raw_bits','input_bits','prefix_bits','sequence_bits','bit_string','base_bits',
            'prefix','suffix_bits','suffix','suffixes','z0','z1','input'}


def bit_values(value):
    if isinstance(value,str) and value and set(value) <= {'0','1'}:
        return [value]
    if isinstance(value,list):
        if value and all(type(v) is int and v in (0,1) for v in value):
            return [''.join(map(str,value))]
        return [b for v in value for b in bit_values(v)]
    if isinstance(value,dict):
        return [b for v in value.values() for b in bit_values(v)]
    return []


def extract(value):
    result=set()
    if isinstance(value,dict):
        bases=[];suffixes=[]
        for key,item in value.items():
            if key in BIT_KEYS:
                bits=bit_values(item);result.update(bits)
                if key in ('bits','raw_bits','input_bits','prefix_bits','base_bits','prefix'):bases.extend(bits)
                if key in ('suffix','suffixes','suffix_bits','z0','z1'):suffixes.extend(bits)
            if isinstance(item,(list,dict)):result.update(extract(item))
        result.update(b+z for b in bases for z in suffixes if len(z)==8)
    elif isinstance(value,list):
        for item in value:
            if isinstance(item,(dict,list)):result.update(extract(item))
    return result


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def seal_data(root,out):
    manifest_path=out/'input_manifest.json'
    if manifest_path.exists():raise RuntimeError('Refusing to regenerate sealed inputs')
    historical=set();historical_training=set();inventory=[];unreadable=[]
    # Retained explicit JSON inputs/records only. No checkpoint/noise deserialization.
    for p in sorted((root/'experiments').rglob('*.json')):
        if p.is_relative_to(out):continue
        try:value=json.loads(p.read_text())
        except (ValueError,UnicodeError) as error:
            unreadable.append(dict(path=str(p.relative_to(root)),reason=str(error)));continue
        strings=extract(value)
        if strings:
            historical.update(strings)
            if 'training' in p.name or p.name.startswith('stage1_task'):
                historical_training.update(strings)
            inventory.append(dict(path=str(p.relative_to(root)),sha256=file_sha(p),explicit_bit_contents=len(strings)))
    training_histories={b[:j] for b in historical_training for j in range(1,len(b))}
    used=set(historical);histories=set(training_histories);rejections={};panels={}
    r=random.Random(6000);training=[]
    for length in (16,24):
        quotas={(a,b):0 for a in (0,1) for b in (0,1)};count=0;reject=0;t=length-8
        while count<512:
            bits=format(r.getrandbits(length),f'0{length}b');stratum=(sum(map(int,bits[:t]))%2,sum(map(int,bits))%2)
            if bits in used or quotas[stratum]>=128:reject+=1;continue
            row=dict(id=f'train/L{length}/row{count:03}',length=length,bits=bits,label=stratum[1],prefix_length=t,
                     prefix_parity=stratum[0],bits_sha256=hashlib.sha256(bits.encode()).hexdigest())
            training.append(row);used.add(bits);quotas[stratum]+=1;count+=1
            histories.update(bits[:j] for j in range(1,length))
        rejections[f'train/L{length}']=reject
    panels['training']=training
    for split,size,seed,suffix_seed in [('development',64,6002,6003),('selection',128,6004,6005),('confirmation',256,6006,6007)]:
        r=random.Random(seed);zgen=random.Random(suffix_seed);rows=[]
        for length in (16,24):
            quotas=[0,0];count=0;reject=0
            while count<size:
                bits=format(r.getrandbits(length),f'0{length}b');label=sum(map(int,bits))%2
                if bits in used or bits[:-1] in histories or quotas[label]>=size//2:reject+=1;continue
                u=format(zgen.getrandbits(7),'07b');suffixes=['0'+u,'1'+u]
                if any(bits+z in used for z in suffixes):reject+=1;continue
                row=dict(id=f'{split}/L{length}/row{count:03}',length=length,bits=bits,label=label,suffixes=suffixes,
                         bits_sha256=hashlib.sha256(bits.encode()).hexdigest(),
                         full_input_hashes=[hashlib.sha256((bits+z).encode()).hexdigest() for z in suffixes])
                rows.append(row);used.update([bits]+[bits+z for z in suffixes]);histories.add(bits[:-1])
                quotas[label]+=1;count+=1
            rejections[f'{split}/L{length}']=reject
        panels[split]=rows
    schedules={}
    for phase,seed in [('common',6001),('paired',6008)]:
        r=random.Random(seed);a=[v['id'] for v in training if v['length']==16];b=[v['id'] for v in training if v['length']==24]
        r.shuffle(a);r.shuffle(b);schedules[phase]=list(map(list,zip(a,b)))
    profiling=[]
    for length in (16,24):
        for parity in (0,1):profiling.extend([v['id'] for v in panels['confirmation'] if v['length']==length and v['label']==parity][:8])
    profile_order=[];r=random.Random(6012)
    # One prospectively interleaved pass per repetition; repetition0 is warm-up.
    for repetition in range(6):
        jobs=[dict(case=case,path=path,repetition=repetition,warmup=repetition==0)
              for case in profiling for path in ['T/P0','T/P1','C/P0','C/P1','F/P0']]
        r.shuffle(jobs);profile_order.extend(jobs)
    features=torch.randn(2,64,generator=torch.Generator().manual_seed(3002),dtype=torch.float32)
    encoded={}
    for split,rows in panels.items():
        for row in rows:
            b=torch.tensor(list(map(int,row['bits'])),dtype=torch.long)
            encoded[row['id']]=dict(x=features[b].unsqueeze(0),labels=b.cumsum(0).remainder(2),
               suffixes=[features[torch.tensor(list(map(int,z)),dtype=torch.long)].unsqueeze(0) for z in row.get('suffixes',[])])
    torch.save(dict(features=features,cases=encoded),out/'sealed_encoded_inputs.pt')
    for split,rows in panels.items():
        (out/f'{split}_inputs.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'update_orders.json').write_text(json.dumps(schedules,indent=2)+'\n')
    (out/'profiling_order.json').write_text(json.dumps(profile_order,indent=2)+'\n')
    manifest=dict(panels={k:dict(path=f'{k}_inputs.json',count=len(v),sha256=file_sha(out/f'{k}_inputs.json'),
                 by_length={str(L):sum(x['length']==L for x in v) for L in (16,24)}) for k,v in panels.items()},
         encoded_sha256=file_sha(out/'sealed_encoded_inputs.pt'),feature_seed=3002,feature_shape=[2,64],
         schedules_sha256=file_sha(out/'update_orders.json'),profile_order_sha256=file_sha(out/'profiling_order.json'),
         generation='Python random.Random(seed).getrandbits(L), zero-padded binary strings; shared input stream per split, lengths16 then24',
         suffix_generation='independent declared suffix stream; draw u only after base/prefix/quota checks; reject whole pair on exact extended-input exposure',
         training_order='one seeded stream shuffles length16 IDs then length24 IDs; zip; one of each per update',
         profiling_selection='first8 cases in each sealed length/parity stratum, original manifest order',profiling_suffix='z0',
         excluded_historical_unique_contents=len(historical),historical_training_contents=len(historical_training),
         rejections=rejections,exposure_inventory=inventory,unreadable_historical_JSON=unreadable,
         limitations=['Unknown Stage1 training exposure','Historical artifacts without explicit retained input bits cannot establish content overlap',
                     'Explicit bits only; encoded-only historical checkpoint/noise files not deserialized'],
         no_outcome_dependent_filtering=True,disjoint_complete_inputs=True,disjoint_incoming_histories=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return panels,schedules,encoded,profile_order
