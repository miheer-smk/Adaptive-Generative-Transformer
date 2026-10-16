"""CPU-only serialized-cache and retained-trace verification; no AGT import."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,gzip,hashlib,collections
from pathlib import Path
import torch
torch.set_grad_enabled(False);torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent
ENG=OUT.parent/'gatec_clrs_insertion_sort_6200_v1'
LOCAL=OUT.parent/'gatec_clrs_pointer_localization_6200_v1'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def tsha(x):return hashlib.sha256(bytes(x.contiguous().flatten().view(torch.uint8).tolist())).hexdigest()
def save(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def main():
    result=json.loads((OUT/'extraction_results_v1.json').read_text());assert result['status']=='COMPLETED_VERIFIED_TRAINING_CACHE'
    cm=json.loads((OUT/'cache_manifest_v1.json').read_text());assert sha(OUT/cm['cache_file'])==cm['sha256']
    cache=torch.load(OUT/cm['cache_file'],map_location='cpu',weights_only=True)
    C,Q,y=cache['candidate_committed_s'],cache['query_committed_s'],cache['predecessor_targets']
    assert C.shape==Q.shape==(128,16,64) and C.dtype==Q.dtype==torch.float32 and y.shape==(128,16) and y.dtype==torch.int64
    assert torch.isfinite(C).all() and torch.isfinite(Q).all() and ((y>=0)&(y<16)).all()
    assert all(x.device.type=='cpu' and not x.requires_grad and x.grad_fn is None for x in [C,Q,y])
    assert torch.equal(cache['row_indices'],torch.arange(128))
    assert torch.equal(cache['candidate_original_element_indices'],torch.arange(16)) and torch.equal(cache['query_original_element_indices'],torch.arange(16))
    inputs=json.loads((OUT/'input_manifest_v1.json').read_text())['arrays']
    summaries=[json.loads(x) for x in (OUT/'per_array_extraction_v1.jsonl').read_text().splitlines()];assert len(summaries)==len(inputs)==128
    rows=json.loads((ENG/'train_pool_v1.json').read_text())[:128]
    refs=[json.loads(x) for x in (ENG/'training_results_v1/training_subset_records_v1.jsonl').read_text().splitlines()];assert len(refs)==128
    for i,(entry,record,row,reference) in enumerate(zip(inputs,summaries,rows,refs)):
        assert cache['array_ids'][i]==entry['id']==record['id']==row['id']==reference['id']==f'official/train/row{i:04d}'
        assert cache['original_input_sha256'][i]==entry['original_input_sha256']==row['original_input_sha256']
        compact=json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        assert hashlib.sha256(compact).hexdigest()==entry['row_content_sha256']
        assert y[i].tolist()==row['pred']==reference['truth']==entry['targets']
        assert tsha(C[i])==record['candidate_tensor_sha256'] and tsha(Q[i])==record['query_tensor_sha256'] and tsha(y[i])==record['target_tensor_sha256']
        assert record['source']==('reused_verified_localization' if i==0 else 'new_frozen_extraction')
    old=torch.load(LOCAL/'final_tensor_evidence_v1.pt',map_location='cpu',weights_only=True)
    assert torch.equal(C[0],old['candidate_states']) and torch.equal(Q[0],old['query_states']) and torch.equal(y[0],old['targets'])
    checkpoint_path=ENG/'training_results_v1/model_update1000_v1.pt'
    assert sha(checkpoint_path)=='2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    Wq=checkpoint['model_state_dict']['query.weight'].double();Wk=checkpoint['model_state_dict']['key.weight'].double()
    scores=(Q.double()@Wq.T)@(C.double()@Wk.T).transpose(1,2)/8
    expected=torch.tensor([r['pointer_logits'] for r in refs],dtype=torch.float64)
    assert torch.allclose(scores,expected,atol=1e-6,rtol=1e-5)
    assert scores.argmax(2).tolist()==[r['predictions'] for r in refs]
    counts=collections.Counter();by_case=collections.defaultdict(list);last_attempt=0
    with gzip.open(OUT/'token_traces_v1.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);assert r['attempt']==last_attempt+1;last_attempt+=1
            assert r['status']=='completed' and r['blocks']==r['selected_actions']==list(range(8))
            assert r['D']==r['P']==r['program_visits']==8 and r['commits']==1 and r['controller_calls']==0
            assert r['parent_unchanged'] and r['returned_state_disjoint']
            t=r['token'];assert r['incoming_metadata']['/tokens_committed']==t and r['returned_metadata']['/tokens_committed']==t+1
            assert not r['incoming_metadata']['/_consumed'] and not r['returned_metadata']['/_consumed']
            c=r['commit'];assert c['token_index']==t and c['entry_count']==7
            assert c['versions_before'].keys()==c['versions_after'].keys()
            assert all(v==t and c['versions_after'][k]==t+1 for k,v in c['versions_before'].items())
            if t==0:
                assert all(v is None for k,v in r['incoming_metadata'].items() if k.endswith(('/committed_keys','/committed_values','/committed_state','/committed_conv_buffer')))
            by_case[r['case']].append(t);counts['tokens']+=1;counts['bank_calls']+=len(r['blocks']);counts['commits']+=1;counts['contextual_entries']+=c['entry_count']
    assert list(by_case)==[f'official/train/row{i:04d}' for i in range(1,128)] and all(v==list(range(32)) for v in by_case.values())
    assert counts['tokens']==counts['commits']==4064 and counts['bank_calls']==32512 and counts['contextual_entries']==28448
    assert result['extraction_elapsed_seconds']<900 and result['parameters_and_buffers_unchanged']
    out=dict(success=True,verified_arrays=128,reused_row0000_bitwise_matches=True,new_arrays=127,
        shape_dtype_and_finiteness=True,graph_free=True,roles_and_original_indices_verified=True,all_input_target_and_payload_hashes_match=True,
        ordinary_head_score_comparison='Existing checkpoint matrices via CPU algebra only; no module forward or fitted parameters',
        max_absolute_pointer_score_error=float((scores-expected).abs().max()),all2048_original_head_predictions_exact=True,
        atol=1e-6,rtol=1e-5,trace_rows=counts['tokens'],trace_arrays=len(by_case),actual_selected_calls=counts['bank_calls'],
        commits=counts['commits'],contextual_entries=counts['contextual_entries'],fresh_initial_cache_metadata_all_arrays=True,
        extraction_elapsed_seconds=result['extraction_elapsed_seconds'],offline_model_constructions=0,offline_model_calls=0,
        offline_backwards=0,offline_optimizer_objects=0,offline_GPU_operations=0,readout_fitting=False,held_out_arrays=0)
    save('cache_verification_v1.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
