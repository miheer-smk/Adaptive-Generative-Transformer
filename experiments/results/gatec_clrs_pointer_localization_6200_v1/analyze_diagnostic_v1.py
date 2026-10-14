"""CPU-only analysis of captured tensors; zero AGT calls or backwards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,gzip,collections,hashlib
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent

def save(name,v):
    (OUT/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def geometry(x):
    x=x.double();norm=x.norm(dim=-1);dist=torch.cdist(x,x);mask=~torch.eye(len(x),dtype=torch.bool)
    return dict(mean_norm=float(norm.mean()),mean_pairwise_L2=float(dist[mask].mean()),max_pairwise_L2=float(dist[mask].max()),
        relative_variation=float(dist[mask].mean()/norm.mean()),min_pairwise_L2=float(dist[mask].min()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    r=json.loads((OUT/'diagnostic_results_v1.json').read_text())
    assert r['status']=='COMPLETED_NO_UPDATE_DIAGNOSTIC'
    result={};model_rows={}
    for name in ['initial','final']:
        data=torch.load(OUT/(name+'_tensor_evidence_v1.pt'),map_location='cpu',weights_only=True)
        raw=data['raw_scores'].double();centered=raw-raw.mean(-1,keepdim=True)
        s=r['models'][name];states=json.loads((OUT/(name+'_committed_state_summaries_v1.json')).read_text())
        assert len(states)==32
        for i,state in enumerate(states):
            assert state['token_index']==i and state['metadata']['/tokens_committed']==i+1 and not state['metadata']['/_consumed']
            assert all(v==i+1 for k,v in state['metadata'].items() if k.endswith('/version'))
        assert torch.equal(data['candidate_states'],data['all_committed_streams'][:16])
        assert torch.equal(data['query_states'],data['all_committed_streams'][16:])
        target=data['targets'];pred=raw.argmax(-1)
        grad=data['gradients']['raw_scores'].double()
        p=torch.softmax(raw,-1)
        analytic=(p-torch.nn.functional.one_hot(target,16))/16
        assert torch.allclose(grad,analytic,atol=1e-6,rtol=1e-5)
        result[name]=dict(CE=s['CE'],pointer_accuracy=s['pointer_accuracy'],predictions=pred.tolist(),targets=target.tolist(),
            mean_target_rank=sum(v['target_rank'] for v in s['queries'])/16,
            mean_target_margin=sum(v['target_margin'] for v in s['queries'])/16,
            mean_logit_span=sum(v['logit_span'] for v in s['queries'])/16,
            mean_max_probability=sum(v['max_probability'] for v in s['queries'])/16,
            mean_entropy=sum(v['entropy'] for v in s['queries'])/16,
            maximum_centered_row_difference=float((centered-centered[:1]).abs().max()),
            centered_score_geometry=geometry(centered),
            query_pipeline=dict(encoded_queries=geometry(data['encoded_tokens'][16:]),prelude_queries=geometry(data['h0'][16:]),
                live_blocks={k:geometry(v[16:]) for k,v in data['block_outputs'].items()},
                committed_queries=geometry(data['query_states']),projected_queries=geometry(data['projected_queries'])),
            candidate_pipeline=dict(encoded_inputs=geometry(data['encoded_tokens'][:16]),prelude_inputs=geometry(data['h0'][:16]),
                live_blocks={k:geometry(v[:16]) for k,v in data['block_outputs'].items()},
                committed_candidates=geometry(data['candidate_states']),projected_keys=geometry(data['projected_keys'])),
            gradient_global_norms={k:float(v.double().norm()) for k,v in data['gradients'].items() if torch.is_tensor(v)},
            gradients_row_norms={k:v.double().norm(dim=-1).tolist() for k,v in data['gradients'].items() if torch.is_tensor(v)},
            parameter_gradient_norms={n:s['parameter_gradient_norms'][n] for n in ['encoder.weight','query.weight','key.weight']},
            index_alignment_pass=True,state_versions_continuity_pass=True,analytic_score_gradient_matches=True)
        model_rows[name]=s
    counts=collections.Counter();models={}
    with gzip.open(OUT/'token_traces_v1.jsonl.gz','rt') as f:
        for line in f:
            x=json.loads(line);counts['attempted']+=1;counts[x['status']]+=1;counts['bank_calls']+=len(x['blocks']);counts['commits']+=x['commits']
            assert x['status']=='completed' and x['D']==x['P']==8 and x['blocks']==x['selected_actions']==list(range(8))
            assert x['parent_unchanged'] and x['returned_state_disjoint']
            xs=models.setdefault(x['phase'],[]);xs.append(x)
    assert dict(counts)==dict(attempted=64,completed=64,bank_calls=512,commits=64)
    for name,xs in models.items():
        assert [x['token'] for x in xs]==list(range(32))
        for prev,nxt in zip(xs,xs[1:]):assert prev['returned_metadata']==nxt['incoming_metadata']
    save('trace_verification_v1.json',dict(success=True,counts=dict(counts),per_model_tokens={k:len(v) for k,v in models.items()},
        continuity_verified=True,canonical_blocks=True,one_commit_per_token=True,zero_extra_model_calls=True))
    save('analysis_v1.json',dict(models=result,unit='Single retained training array, two model identities',
        no_inferential_uncertainty=True,no_model_calls=True,no_backward=True,
        state_adjoint_scope='Original total-episode CE, including later query/state paths; not isolated per-query causal gradients',
        projected_key_adjoint_scope='Combined loss credit from all queries; candidate-state adjoints also include recurrent/future paths',
        target_rank_definition='1 plus number of scores strictly greater than correct-target score; ties explicitly retained'))
    a,b=result['initial'],result['final']
    lines=['# CLRS pointer-failure localization, seed6200',
        '\nStatus: **COMPLETED_NO_UPDATE_DIAGNOSTIC**. Historical training remains **COMPLETED_NON_SUCCESS**.',
        '\nOne retained input: `official/train/row0000`, size16. Fresh initialization exactly matches the recorded identity; update1000 checkpoint loads strictly and matches its recorded final parameter identity. Both forwards match retained original scores and CE at atol1e-6/rtol1e-5.',
        '\n|Measured stage: mean pairwise L2|Initial|Final|','|---|---:|---:|']
    for label,k in [('Committed query states','query_states'),('Committed candidates','candidate_states'),('Projected queries','projected_queries'),('Projected keys','projected_keys')]:
        lines.append(f'|{label}|{model_rows["initial"]["stages"][k]["mean_pairwise_L2"]:.9g}|{model_rows["final"]["stages"][k]["mean_pairwise_L2"]:.9g}|')
    lines.extend([f'\nCE initial/final: {a["CE"]:.12g}/{b["CE"]:.12g}. Both predict candidate13 for every query and get0/16 correct; candidate13 is this array’s maximum and never a predecessor target.',
        f'\nMean logit span initial/final: {a["mean_logit_span"]:.9g}/{b["mean_logit_span"]:.9g}. Mean maximum probability: {a["mean_max_probability"]:.9g}/{b["mean_max_probability"]:.9g}; uniform is0.0625. Mean target rank: {a["mean_target_rank"]}/{b["mean_target_rank"]}. Full raw/centered scores, probabilities, targets, ranks, margins and adjoints are retained.',
        '\nThe final model retains distinct encoded query inputs and prelude states. Selected live-block norms grow sharply, while relative query variation contracts. Query-token mean norm reaches about1893 before terminal normalization; after the ordinary commit the mean norm is7.704 and mean pairwise query distance is0.003237. Candidate states show the same substantial contraction.',
        '\nThis localizes the major measured loss of distinctions upstream of the task projections, along the selected-block/terminal-normalization path. It is a measured association with a dominant shared live-state component, not proof that a particular operator or commit causes the historical optimization failure. The projections do not map distinct inputs to exact equality and preserve/increase relative variation in this case.',
        '\nAll retained candidate/query/projected-query/projected-key task-loss adjoints are connected, finite and nonzero. Directions vary across elements/queries. The final projected-query mean adjoint norm is much smaller than initially; score-level CE gradients remain substantial and match the analytic CE derivative. This demonstrates attenuated connected credit rather than a severed graph. It does not establish useful historical gradient alignment.',
        '\nQuery-state adjoints include later recurrent computation. Projected-key adjoints combine all query losses. Do not interpret these as independent local intervention estimates.',
        '\n## Scope and limits',
        '\nOne array and two checkpoints do not locate the historical onset, prove the gradient spikes caused collapse, establish operator causality or generalize this mechanism to other inputs. Initial scores are already weak and incorrect; training reduced distinction substantially here without changing the repeated argmax. No training, optimizer construction, update, warm-up, retry or third rollout occurred.',
        '\n## Exact ledger','```json',json.dumps(r['ledger'],indent=2),'```',
        '\n## Integrity','```json',json.dumps(r['integrity'],indent=2),'```',
        '\nAll parameter/buffer identities are unchanged across each backward. Full tensor evidence is in `initial_tensor_evidence_v1.pt` and `final_tensor_evidence_v1.pt`; descriptive measurements are in `analysis_v1.json`, per-query records in the model result files, and metadata/state summaries in model-specific committed-state files. Publication receipts are added separately.'])
    (OUT/'report_v1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
