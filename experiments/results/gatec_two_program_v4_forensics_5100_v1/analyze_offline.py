"""One standard-library-only descriptive pass over immutable v4 records.

No model imports, checkpoint deserialization, execution, bootstrap or inference.
Every output is opened exclusively; this script cannot overwrite or resume.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
INPUT = ROOT/'experiments/results/gatec_two_program_feasibility_5100_v4'
ATOL, RTOL = 1e-6, 1e-5


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(name, value):
    with (OUT/name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def read_json(name):
    with (INPUT/name).open('r') as handle:
        return json.load(handle)


def shape(value):
    if not isinstance(value, list):
        return ()
    children = [shape(child) for child in value]
    if children and any(s != children[0] for s in children):
        return None
    return (len(value),) + (children[0] if children else ())


def flat(value):
    if isinstance(value, list):
        for child in value:
            yield from flat(child)
    else:
        yield value


def arrays(left, right):
    if shape(left) != shape(right) or shape(left) is None:
        return dict(available=True, shape_match=False, left_shape=shape(left), right_shape=shape(right), allclose=False)
    a, b = list(flat(left)), list(flat(right))
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in a+b):
        return dict(available=False, reason='not finite numeric arrays')
    delta = [y-x for x,y in zip(a,b)]
    squares = math.fsum(d*d for d in delta)
    return dict(available=True, representation='Retained full array', shape=list(shape(left)),
        element_count=len(a), exact_values_equal=a==b,
        allclose=all(abs(x-y) <= ATOL+RTOL*abs(y) for x,y in zip(a,b)),
        max_abs_difference=max(map(abs,delta), default=0.),
        L2_of_actual_difference=math.sqrt(squares),
        RMS_of_actual_difference=math.sqrt(squares/len(delta)) if delta else 0.,
        signed_mean_difference=math.fsum(delta)/len(delta) if delta else 0.)


def close_scalar(a, b):
    if isinstance(a,float) and isinstance(b,(float,int)) and not isinstance(b,bool):
        return math.isclose(a,b,abs_tol=ATOL,rel_tol=RTOL)
    return type(a) is type(b) and a==b


def tree_equal(a, b):
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return isinstance(b,list) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return close_scalar(a,b)


def summaries(left, right):
    """Signed changes of retained summaries, never distances of raw states."""
    rows=[]
    for key in dict.fromkeys(list(left)+list(right)):
        if key not in left or key not in right:
            rows.append(dict(field=key, representation='Missing/unavailable',
                left_present=key in left, right_present=key in right, comparable=False))
            continue
        a,b=left[key],right[key]
        row=dict(field=key,left=a,right=b,locked_match=tree_equal(a,b),comparable=True)
        if isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b)=={'shape','l2','max_abs'}:
            row.update(representation='Retained summary', shape_equal=a['shape']==b['shape'],
                summary_l2_delta=b['l2']-a['l2'],summary_max_abs_delta=b['max_abs']-a['max_abs'],
                caution='Differences of published norms/maxima, not a tensor distance or state equality proof')
        else:
            row['representation']='Exact metadata' if a is None or not isinstance(a,float) else 'Retained summary'
        rows.append(row)
    return rows


def arithmetic(summary):
    logits=summary['logits'][0]
    top=max(logits)
    computed=top+math.log(math.fsum(math.exp(z-top) for z in logits))-logits[summary['label']]
    probabilities=[math.exp(z-top)/math.fsum(math.exp(w-top) for w in logits) for z in logits]
    return dict(recorded_CE=summary['CE'],arithmetic_CE=computed,
        CE_residual=summary['CE']-computed,CE_locked_scalar_match=close_scalar(summary['CE'],computed),
        recorded_prediction=summary['prediction'],argmax_from_recorded_logits=logits.index(top),
        prediction_consistent=summary['prediction']==logits.index(top),
        correctness_consistent=summary['correct']==(summary['prediction']==summary['label']),
        probability_arrays=arrays(summary['probabilities'],[probabilities]),
        caution='Arithmetic corroboration only. Published losses and predictions remain authoritative.')


def semantic_key(record):
    case=record['case']
    return (case['phase'],case.get('sequence_id'),case.get('history_program'),
            case.get('program'),case.get('arm'),record['token_index'],record['program'],record['reference'])


def schema(value):
    if isinstance(value,dict):
        return {k:schema(v) for k,v in value.items()}
    if isinstance(value,list):
        if all(type(v) in (bool,int,float) for v in flat(value)):
            return dict(representation='JSON array',shape=shape(value),leaf_types=sorted({type(v).__name__ for v in flat(value)}))
        return dict(representation='JSON list of records',length=len(value),
            element_schemas=list({json.dumps(schema(v),sort_keys=True):schema(v) for v in value}.values()))
    return type(value).__name__


def csv_table(name, rows, columns):
    with (OUT/name).open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items() if k in columns})


def execute():
    if (OUT/'analysis_started.json').exists():
        raise SystemExit('STOP: this one offline analysis has already started; no resume')
    before=json.loads((OUT/'integrity_before.json').read_text())
    plan=json.loads((OUT/'analysis_plan.json').read_text())
    changed=[name for name,info in before['evidence_files'].items() if sha(INPUT/name)!=info['sha256']]
    if changed:
        raise SystemExit(f'STOP: evidence changed before analysis: {changed}')
    write_json('analysis_started.json',dict(plan_sha256=sha(OUT/'analysis_plan.json'),
        script_sha256=sha(Path(__file__)),mode='standard-library-only offline arithmetic',
        source='v4 only',no_model_import=True,no_checkpoint_deserialization=True))
    selected=plan['selection']['controls']+[plan['selection']['primary']]
    ledger=read_json('execution_ledger.json')
    all_current=read_json('current_cases.json')
    pairs=read_json('paired_comparisons.json')
    stored_analysis=read_json('current_screen_analysis.json')
    verification=read_json('final_verification.json')
    failure=read_json('failure.json')
    archive_gate=read_json('archival_gate.json')
    current_by_id=defaultdict(list)
    for row in all_current:
        current_by_id[(row['case']['sequence_id'],row['case']['history_program'],row['case']['program'])].append(row)
    records=defaultdict(list)
    global_counts=Counter(); phases=Counter(); blocks=Counter();P1_counts=Counter()
    selected_traces=[];schema_variants={};ambiguities=[]
    with gzip.open(INPUT/'token_traces.jsonl.gz','rt') as handle:
        for line in handle:
            record=json.loads(line)
            global_counts['tokens']+=1
            global_counts['commits']+=record['commit_count']
            global_counts['decisions']+=record['D']
            global_counts['executions']+=sum(len(d['actual_blocks']) for d in record['decisions'])
            global_counts['restart']+=record['restart_count'];global_counts['repeat']+=record['repeat_count']
            phases[(record['case']['phase'],record['program'],record['reference'])]+=1
            for d in record['decisions']:
                blocks.update(d['actual_blocks'])
            if record['program']=='P1':
                P1_counts['tokens']+=1;P1_counts[record['case']['phase']]+=1
                P1_counts['block5']+=sum(d['actual_blocks'].count(5) for d in record['decisions'])
                P1_counts['repeat_actions']+=sum(d['action']==8 for d in record['decisions'])
            if record['case'].get('sequence_id') in selected:
                # Record association is explicit; physical array order is not a decision identity.
                ids=[row['decision'] for row in record['decisions']]
                if len(ids)!=len(set(ids)) or set(ids)!=set(range(record['D'])):
                    ambiguities.append(dict(identity=semantic_key(record),decision_ids=ids,reason='Missing/duplicate explicit decision ID; skip trajectory joins'))
                    continue
                records[semantic_key(record)].append(record)
                selected_traces.append(record)
                signature=json.dumps(schema(record),sort_keys=True)
                schema_variants[signature]=schema(record)
    for key,values in records.items():
        if len(values)!=1:ambiguities.append(dict(key=key,count=len(values)))

    def token(sid,phase,hist,program,index,reference):
        case_program=program if phase=='current' else None
        key=(phase,sid,hist,case_program,None,index,f'P{program}',reference)
        rows=records.get(key,[])
        if len(rows)!=1:
            ambiguities.append(dict(query=key,matches=len(rows)))
            return None
        return rows[0]

    def current(sid,hist,prog):
        rows=current_by_id.get((sid,hist,prog),[])
        if len(rows)!=1:
            if len(rows)>1:ambiguities.append(dict(current_query=[sid,hist,prog],matches=len(rows)))
            return None
        return rows[0]

    def at(record, decision_id):
        candidates=[r for r in record['decisions'] if r['decision']==decision_id]
        if len(candidates)!=1:
            raise ValueError('Unexpected unresolved explicit decision association')
        return candidates[0]

    def terminal_row(record):
        # Last observed explicit decision; not an inferred token grouping.
        return at(record,max(r['decision'] for r in record['decisions']))

    def returned_observation(sid,hist,t,record):
        if t<14:
            nxt=token(sid,'prefix',hist,hist,t+1,hist==0)
        else:
            nxt=token(sid,'current',hist,0,15,False)
        if nxt is None:return dict(available=False,reason='No unique next retained observation')
        match=record['outgoing_state_sha256']==nxt['incoming_state_sha256']
        if not match:return dict(available=False,reason='State fingerprint association mismatch')
        return dict(available=True,source=dict(phase=nxt['case']['phase'],sequence_id=sid,
            history_program=hist,token_index=nxt['token_index'],boundary_program=0 if t==14 else hist),
            outgoing_incoming_fingerprint_equal=True,
            summary=at(nxt,0)['persistent'],
            caution='Retained next-input summary linked by state fingerprint; not a direct commit-time full tensor.')

    four_cells=[];all_checks=[];canonical_validation=[]
    for sid in selected:
        canon=current(sid,0,0)
        for hist in [0,1]:
            for prog in [0,1]:
                cell=current(sid,hist,prog)
                if cell is None:
                    four_cells.append(dict(sequence_id=sid,history_program=hist,boundary_program=prog,
                        status='UNOBSERVED',reason='No executed boundary record; no reconstruction'))
                    continue
                summary=cell['summary'];trace=token(sid,'current',hist,prog,15,False)
                fields=['label','prediction','CE','logits','probabilities','P','D','commit_count',
                        'executed_blocks','incoming_state_sha256','outgoing_state_sha256']
                match=trace is not None and all(tree_equal(summary[f],trace[f]) for f in fields)
                arith=arithmetic(summary);all_checks.append(arith)
                expected_canon=canon['summary'] if canon else None
                canonical_match=expected_canon is not None and cell['canonical_same_input']['case']==expected_canon['case']
                canonical_validation.append(dict(sequence_id=sid,hist=hist,prog=prog,
                    canonical_identity_matches=canonical_match,
                    canonical_record_values_match=expected_canon is not None and tree_equal(cell['canonical_same_input'],expected_canon)))
                four_cells.append(dict(sequence_id=sid,history_program=hist,boundary_program=prog,
                    status='EXECUTED_ACCEPTED' if cell['acceptance']['accepted'] else 'EXECUTED_REJECTED',
                    label=summary['label'],prediction=summary['prediction'],CE=summary['CE'],logits=summary['logits'],
                    correct=summary['correct'],P=summary['P'],D=summary['D'],commit_count=summary['commit_count'],
                    blocks=summary['executed_blocks'],incoming_sha256=summary['incoming_state_sha256'],
                    outgoing_sha256=summary['outgoing_state_sha256'],trace_summary_matches=match,
                    published_acceptance=cell['acceptance'],arithmetic=arith))
    pair_checks=[]
    for pair in pairs:
        if pair['sequence_id'] not in selected:continue
        sid=pair['sequence_id'];hist=pair['history_program']
        a,b=current(sid,hist,0),current(sid,hist,1)
        association=a is not None and b is not None and pair['horizon']==0
        if association:
            association=all(tree_equal(pair[name],cell['summary']) for name,cell in [('P0',a),('P1',b)])
            computed=b['summary']['CE']-a['summary']['CE']
            tv=.5*math.fsum(abs(x-y) for x,y in zip(flat(a['summary']['probabilities']),flat(b['summary']['probabilities'])))
            pair_checks.append(dict(sequence_id=sid,history_program=hist,association_confirmed=association,
                recorded_CE_delta=pair['CE_delta'],arithmetic_CE_delta=computed,
                recorded_TV=pair['total_variation'],arithmetic_TV=tv,
                CE_delta_match=close_scalar(pair['CE_delta'],computed),TV_match=close_scalar(pair['total_variation'],tv),
                same_incoming_fingerprint=a['summary']['incoming_state_sha256']==b['summary']['incoming_state_sha256']))
        else:pair_checks.append(dict(sequence_id=sid,history_program=hist,association_confirmed=False))
    with (INPUT/'current_comparison_table.csv').open(newline='') as handle:
        published_csv=[row for row in csv.DictReader(handle) if row['sequence_id'] in selected]
    csv_checks=[]
    for row in published_csv:
        cell=current(row['sequence_id'],int(row['history_program']),int(row['boundary_program']))
        csv_checks.append(dict(identity=[row['sequence_id'],row['history_program'],row['boundary_program']],
            matches=cell is not None and int(row['label'])==cell['summary']['label']
                and int(row['prediction'])==cell['summary']['prediction']
                and close_scalar(float(row['CE']),cell['summary']['CE'])))

    timeline=[];array_rows=[];summary_rows=[];firsts={};boundary_components={};continuity=[];metadata_checks=[]
    missing=[]
    for sid in selected:
        first_program_difference=None;first_full_array_difference=None;first_prefix_output_disagreement=None
        for t in range(15):
            a=token(sid,'prefix',0,0,t,True);b=token(sid,'prefix',1,1,t,False)
            if a is None or b is None:
                missing.append(dict(sequence_id=sid,token=t,scope='prefix'));continue
            if a['label']!=b['label']:ambiguities.append(dict(sequence_id=sid,token=t,reason='Input-prefix labels differ'))
            arith_a,arith_b=arithmetic(a),arithmetic(b);all_checks.extend([arith_a,arith_b])
            returned_a,returned_b=returned_observation(sid,0,t,a),returned_observation(sid,1,t,b)
            continuity.append(dict(sequence_id=sid,token=t,P0={k:v for k,v in returned_a.items() if k!='summary'},
                P1={k:v for k,v in returned_b.items() if k!='summary'}))
            stage_summaries={'incoming':summaries(at(a,0)['persistent'],at(b,0)['persistent'])}
            if returned_a['available'] and returned_b['available']:
                stage_summaries['returned_next_input_observation']=summaries(returned_a['summary'],returned_b['summary'])
            for stage,rows in stage_summaries.items():
                for row in rows:summary_rows.append(dict(sequence_id=sid,phase='prefix',token=t,stage=stage,**row))
            prep={k:arrays(at(a,0)[k],at(b,0)[k]) for k in ['h_before','h0','memory_read','memory_mean']}
            for k,result in prep.items():array_rows.append(dict(sequence_id=sid,phase='prefix',token=t,decision=0,stage='preparation',field=k,**result))
            if a['outgoing_state_sha256']!=b['outgoing_state_sha256']:
                outgoing_fp_match=False
            else:outgoing_fp_match=True
            first_decision_difference=None
            for d in range(8):
                ar,br=at(a,d),at(b,d)
                if ar['decision']!=d or br['decision']!=d:
                    ambiguities.append(dict(sequence_id=sid,token=t,decision=d,reason='Explicit decision identities differ'))
                intended_same=ar['action']==br['action'] and ar['actual_blocks']==br['actual_blocks']
                if not intended_same and first_program_difference is None:
                    first_program_difference=dict(token=t,decision=d,P0_action=ar['action'],P1_action=br['action'],
                        P0_blocks=ar['actual_blocks'],P1_blocks=br['actual_blocks'],interpretation='Prescribed compound program substitution, not a defect')
                for field in ['h_before','h_after','h0','memory_read','memory_mean']:
                    metric=arrays(ar[field],br[field]);array_rows.append(dict(sequence_id=sid,phase='prefix',token=t,
                        decision=d,stage='live_execution',field=field,same_action_and_block=intended_same,**metric))
                    if metric.get('allclose') is False:
                        point=dict(token=t,decision=d,field=field,comparison=metric,same_action_and_block=intended_same)
                        if first_full_array_difference is None:first_full_array_difference=point
                        if first_decision_difference is None:first_decision_difference=point
                discrete_fields=['execution_before','execution_after','pointer_before','pointer_after',
                    'action','actual_blocks','just_restarted_before','just_restarted_after','legal_mask']
                metadata_checks.append(dict(sequence_id=sid,phase='prefix',token=t,decision=d,
                    differences={k:dict(P0=ar[k],P1=br[k]) for k in discrete_fields if ar[k]!=br[k]},
                    interpretation='Action6 substitution and its pointer/mask consequences are intended; P stays8'))
            if a['prediction']!=b['prediction'] and first_prefix_output_disagreement is None:
                first_prefix_output_disagreement=dict(token=t,label=a['label'],P0_prediction=a['prediction'],P1_prediction=b['prediction'])
            terminal=arrays(terminal_row(a)['h_after'],terminal_row(b)['h_after'])
            logits=arrays(a['logits'],b['logits'])
            array_rows.extend([dict(sequence_id=sid,phase='prefix',token=t,decision=7,stage='terminal',field='h_terminal_from_last_h_after',**terminal),
                               dict(sequence_id=sid,phase='prefix',token=t,decision=None,stage='output',field='logits',**logits)])
            timeline.append(dict(sequence_id=sid,token=t,human_token=t+1,label=a['label'],
                P0_prediction=a['prediction'],P1_prediction=b['prediction'],P0_CE=a['CE'],P1_CE=b['CE'],
                incoming_fingerprint_equal=a['incoming_state_sha256']==b['incoming_state_sha256'],
                returned_fingerprint_equal=outgoing_fp_match,
                incoming_summary_matching_fields=sum(r['locked_match'] for r in stage_summaries['incoming']),
                incoming_summary_field_count=len(stage_summaries['incoming']),preparation=prep,
                first_live_array_difference_this_token=first_decision_difference,terminal=terminal,output_logits=logits,
                readout_role='Intermediate prefix readout, not a retrospective acceptance requirement'))

        a=token(sid,'current',0,0,15,False);b=token(sid,'current',1,0,15,False)
        ca,cb=current(sid,0,0),current(sid,1,0)
        if any(x is None for x in [a,b,ca,cb]):
            missing.append(dict(sequence_id=sid,scope='final P0 boundary'));continue
        stages={'incoming':(at(a,0)['persistent'],at(b,0)['persistent']),
                'commit_pre':(ca['summary']['archival_commit']['commit_pre'],cb['summary']['archival_commit']['commit_pre']),
                'committed_return':(ca['summary']['archival_commit']['commit'],cb['summary']['archival_commit']['commit'])}
        boundary_stage={}
        for stage,(left,right) in stages.items():
            rows=summaries(left,right);boundary_stage[stage]=rows
            for row in rows:summary_rows.append(dict(sequence_id=sid,phase='final_P0_boundary',token=15,stage=stage,**row))
        incoming_first=next((r for r in boundary_stage['incoming'] if not r['locked_match']),None)
        live=[];first_boundary_array=None
        for d in range(8):
            ar,br=at(a,d),at(b,d)
            discrete={k:dict(canonical_history=ar[k],P1_history=br[k]) for k in
                ['action','actual_blocks','pointer_before','pointer_after','execution_before','execution_after',
                 'legal_mask','just_restarted_before','just_restarted_after'] if ar[k]!=br[k]}
            metadata_checks.append(dict(sequence_id=sid,phase='final_P0_boundary',token=15,decision=d,differences=discrete))
            for field in ['h_before','h_after','h0','memory_read','memory_mean']:
                metric=arrays(ar[field],br[field]);entry=dict(sequence_id=sid,phase='final_P0_boundary',
                    token=15,decision=d,stage='same_program_live_execution',field=field,**metric)
                live.append(entry);array_rows.append(entry)
                if first_boundary_array is None and metric.get('allclose') is False:first_boundary_array=entry
        commit_terminal=arrays(ca['summary']['archival_commit']['terminal'],cb['summary']['archival_commit']['terminal'])
        array_rows.append(dict(sequence_id=sid,phase='final_P0_boundary',token=15,decision=None,stage='terminal_commit',field='terminal',**commit_terminal))
        for name,c in [('P0_history',ca),('P1_history',cb)]:
            tr=a if name=='P0_history' else b
            continuity.append(dict(sequence_id=sid,history=name,token=15,
                terminal_snapshot_matches_last_live_array=arrays(c['summary']['archival_commit']['terminal'],terminal_row(tr)['h_after']),
                commit_pre_matches_incoming_summary=tree_equal(c['summary']['archival_commit']['commit_pre'],at(tr,0)['persistent'])))
        supplied=token(sid,'supplied_reference',1,0,15,True)
        supplied_comparisons=[]
        if supplied:
            for d in range(8):
                for field in ['h_before','h_after','h0','memory_read','memory_mean']:
                    supplied_comparisons.append(dict(decision=d,field=field,**arrays(at(b,d)[field],at(supplied,d)[field])))
            supplied_comparisons.append(dict(field='logits',**arrays(b['logits'],supplied['logits'])))
        boundary_components[sid]=dict(summary_stages=boundary_stage,live_arrays=live,
            terminal_full_array=commit_terminal,first_retained_differing_incoming_summary=incoming_first,
            first_retained_differing_full_live_array=first_boundary_array,
            output_logits=arrays(a['logits'],b['logits']),
            exact_return_metadata_equal=a['state_metadata']==b['state_metadata'],
            supplied_P0_reference_same_P1_history={'available':supplied is not None,
                'array_comparisons':supplied_comparisons,
                'outgoing_fingerprint_equal':supplied['outgoing_state_sha256']==b['outgoing_state_sha256'] if supplied else None,
                'note':'Same supplied P1-generated state, not fully canonical-history task target. Full state comparison is a reported runtime outcome, not retained full tensors here.'})
        firsts[sid]=dict(first_prefix_program_difference=first_program_difference,
            first_prefix_full_array_difference=first_full_array_difference,
            first_prefix_prediction_disagreement=first_prefix_output_disagreement,
            final_P0_boundary_first_incoming_summary_difference=incoming_first,
            final_P0_boundary_first_full_live_array_difference=first_boundary_array,
            caution='Observation order and numerical association do not identify the first causal error or causal component.')

    # Exact metadata and observable within-token constancy; no claim of a new purity experiment.
    protocol_checks=[]
    for tr in selected_traces:
        if tr['case']['phase'] not in ['prefix','current','supplied_reference']:continue
        first=at(tr,0);all_rows=tr['decisions'];meta=tr['state_metadata']
        version_keys=[k for k in meta if k.endswith('/version')]
        protocol_checks.append(dict(identity=semantic_key(tr),P=tr['P'],D=tr['D'],commits=tr['commit_count'],
            selected_block_count=sum(len(r['actual_blocks']) for r in all_rows),
            retained_incoming_summaries_constant=all(tree_equal(first['persistent'],r['persistent']) for r in all_rows),
            retained_context_summaries_constant=all(tree_equal(first['context'],r['context']) for r in all_rows),
            anchor_full_array_exactly_constant=all(first['h0']==r['h0'] for r in all_rows),
            memory_read_full_array_exactly_constant=all(first['memory_read']==r['memory_read'] for r in all_rows),
            returned_token_count=meta['/tokens_committed'],returned_consumed=meta['/_consumed'],
            recorded_cache_versions_increment_once=all(meta[k]==first['persistent'][k]+1 for k in version_keys),
            window_metadata={k:v for k,v in meta.items() if 'window' in k or 'num_committed_tokens' in k},
            actual_rotary_positions='Missing/unavailable: no independently retained positional-index array',
            caution='Descriptive consistency of retained observations; prior full purity/commit assertions carried as runtime evidence, not newly rerun.'))

    expected=dict(tokens=1979,executions=15832,commits=1979)
    counts_ok=all(global_counts[k]==v for k,v in expected.items())
    ledger_counts_ok=len(ledger['calls'])==1979 and all(c['status']=='COMPLETED' and c['actual_bank_calls']==8 and c['actual_commits']==1 for c in ledger['calls'])
    block_counts=dict(sorted(blocks.items()))
    reconciled=dict(observed=dict(global_counts),published_expected=expected,counts_match=counts_ok,
        ledger_calls_match=ledger_counts_ok,
        phase_program_reference=[dict(phase=p,program=g,reference=r,tokens=n) for (p,g,r),n in sorted(phases.items())],
        P1=dict(P1_counts),block_execution_counts=block_counts,
        block5_extra_over_one_per_token=blocks[5]-global_counts['tokens'],
        P1_extra_block5_corresponds_to_67_tokens=P1_counts['tokens']==67 and P1_counts['prefix']==60
            and P1_counts['current']==7 and P1_counts['block5']==134 and blocks[5]-global_counts['tokens']==67,
        block6_omissions=global_counts['tokens']-blocks[6],
        future_calls_recorded=ledger['future_calls'],future_records_observed=sum(n for (p,g,r),n in phases.items() if p=='future'),
        joint_bootstrap_record='Not executed, recorded in final_report/results; no joint_uncertainty.json exists',
        joint_uncertainty_artifact_present=(INPUT/'joint_uncertainty.json').exists(),
        historical_work='Separate published historical ledgers, not pooled',
        caution='Arithmetic corroboration of retained traces and counters; not new implementation validation.')

    coverage=[
        dict(field='h_before,h_after,h0,memory_read,memory_mean',representation='Retained full array',
            boundaries='Each selected prefix/boundary bank decision',comparison='Actual corresponding elementwise differences'),
        dict(field='pointer_one_hot,budget,script-provider logits',representation='Retained full array',
            boundaries='Adapter/controller observations only; unavailable on reference-hook prefix traces',
            caveat='Script logits are not learned-controller or seed3401 monitor logits. No normalized395-vector for these selected cases.'),
        dict(field='output logits and probabilities',representation='Retained full array',
            boundaries='Completed prefix and boundary outputs',caveat='Published outputs, not model recomputation'),
        dict(field='prediction,label,correct',representation='Exact metadata',
            boundaries='Completed prefix and boundary outputs',caveat='Prefix readouts are descriptive, not new acceptance gates'),
        dict(field='CE',representation='Retained summary',
            boundaries='Published scalar loss after each completed token',caveat='Arithmetic from recorded logits is corroboration only'),
        dict(field='persistent/context s,M,GDN state/buffer,KV/SWA keys/values',representation='Retained summary',
            boundaries='Bank-observer incoming snapshots, and current-case commit_pre/commit',
            caveat='Only shape/l2/max_abs; no elementwise state distance or equality from equal summaries'),
        dict(field='terminal at current boundary',representation='Retained full array',
            boundaries='current_cases.summary.archival_commit.terminal; cross-check last h_after'),
        dict(field='prefix terminal',representation='Retained full array',
            boundaries='Last bank h_after; no separately retained prefix commit terminal record'),
        dict(field='prefix commit_pre/commit',representation='Missing/unavailable',
            boundaries='Direct commit-time JSON snapshots absent; observer appends archival_commit after writing token trace',
            alternative='Next-token incoming summary may be linked via exact recorded state fingerprint'),
        dict(field='returned-state fields: dtype/shape/cache versions/window counts/tokens_committed/_consumed',representation='Exact metadata',
            boundaries='Every token trace state_metadata',caveat='No explicit independent rotary-position values; logical token/window counters only'),
        dict(field='incoming/outgoing state SHA256',representation='Exact metadata',
            boundaries='Every completed token',caveat='Fingerprints can bind observations; do not expose raw state arrays'),
        dict(field='full state equality/purity/storage isolation/commit payload comparisons',
            representation='Runtime comparison outcome without retained underlying tensors',
            boundaries='Archival comparison flags; published gates/reports; adapter assertion path',
            caveat='No new full-state equality test; selected supplied-reference agreement is reconstructed descriptively only for retained live arrays and fingerprints'),
        dict(field='serialized state bundles committed_states_00/01/02.pt',representation='Missing/unavailable for this analysis',
            boundaries='Files exist; deliberately not opened/deserialized; v4 code says they contain incoming/root clones',
            caveat='Not claimed absent on disk; their actual serialized tensor schema is not inspected. Row03 bundle was never created.'),
        dict(field='GDN temporary state,gate/projection internals,full memory slots/raw committed states',representation='Missing/unavailable in JSON',
            boundaries='Not retained as full arrays in selected readable records',caveat='Cannot identify a semantic bit, causal component or information loss from summaries'),
        dict(field='row03 P1→P1 and all suffix continuations',representation='Missing/unavailable',
            boundaries='Not executed; no reconstruction or interpolation permitted'),
    ]
    with (INPUT/'archival_comparisons.jsonl').open() as handle:
        archival=[json.loads(line) for line in handle]
    schemas=dict(selected_trace_variants=list(schema_variants.values()),selected_trace_count=len(selected_traces),
        selected_current_schema_variants=list({json.dumps(schema(c),sort_keys=True):schema(c) for c in all_current if c['case']['sequence_id'] in selected}.values()),
        archival_schema_variants=list({json.dumps(schema(a),sort_keys=True):schema(a) for a in archival}.values()),
        coverage=coverage,join_ambiguities=ambiguities,missing=missing,
        join_definition=plan['joins'],selected_ids=selected,token_index_rule='Use explicit token_index/decision fields; no //8 or %8 reconstruction')
    consistency=dict(current_trace_crosschecks=all(c.get('trace_summary_matches',True) for c in four_cells),
        canonical_comparator_crosschecks=canonical_validation,paired_values=pair_checks,published_csv=csv_checks,
        logits_CE_prediction_arithmetic=all_checks,
        all_CE_checks_within_locked_scalar_tolerance=all(a['CE_locked_scalar_match'] for a in all_checks),
        all_prediction_checks=all(a['prediction_consistent'] and a['correctness_consistent'] for a in all_checks),
        primary_identity_matches_failure=any(c['sequence_id']==plan['selection']['primary'] and c.get('status')=='EXECUTED_REJECTED' for c in four_cells),
        failure_boundary= failure['boundary'],stored_analysis_primary_case=stored_analysis['failure']['case'],
        archived_comparison_records=len(archival),archival_control_status=archive_gate['status'])

    write_json('schema_and_coverage.json',schemas)
    write_json('four_cell_comparisons.json',four_cells)
    write_json('paired_timelines.json',timeline)
    write_json('array_differences.json',array_rows)
    write_json('component_summary_differences.json',summary_rows)
    write_json('boundary_component_comparisons.json',boundary_components)
    write_json('first_observed_differences.json',firsts)
    write_json('state_continuity_and_commit_coverage.json',continuity)
    write_json('recorded_metadata_comparisons.json',metadata_checks)
    write_json('retained_protocol_consistency.json',protocol_checks)
    write_json('ledger_reconciliation.json',reconciled)
    write_json('arithmetic_and_identity_crosschecks.json',consistency)
    csv_table('four_cell_table.csv',four_cells,['sequence_id','history_program','boundary_program','status','label','prediction','CE','correct','P','D','commit_count','blocks','logits'])
    csv_table('array_differences.csv',array_rows,['sequence_id','phase','token','decision','stage','field','shape','allclose','exact_values_equal','max_abs_difference','L2_of_actual_difference','RMS_of_actual_difference','same_action_and_block'])
    csv_table('component_summary_differences.csv',summary_rows,['sequence_id','phase','token','stage','field','representation','locked_match','shape_equal','summary_l2_delta','summary_max_abs_delta','left','right'])

    report='# Offline v4 forensic analysis: recorded switching failure\n\n'
    report+='This standard-library-only analysis used existing v4 evidence. No model or checkpoint was loaded, no forward/backward or test executed, and no future or missing boundary was reconstructed. No bootstrap or new inferential analysis was performed.\n\n'
    report+='The case scope was sealed before derived measurements: primary acquisition/L16/row03; controls row00,row01,row02, identified by the preceding accepted P1→P0 calls in retained execution order. These partial ordered cases are not full-panel P1 performance estimates. V4 still rejects P1 at its current/switch contract.\n\n'
    report+='## Matched four-cell observations\n\n|Case|History → boundary|Recorded truth|Prediction|Recorded CE|V4 acceptance|\n|---|---|---:|---:|---:|---|\n'
    for cell in four_cells:
        label=cell.get('label','—');prediction=cell.get('prediction','—');ce=f"{cell['CE']:.12g}" if 'CE' in cell else '—'
        report+=f"|{cell['sequence_id'].split('/')[-1]}|P{cell['history_program']} → P{cell['boundary_program']}|{label}|{prediction}|{ce}|{cell['status']}|\n"
    report+='\nRow03 P0→P1 is a genuinely retained same-input boundary, not inferred. Row03 P1→P1 is unobserved. Same-input canonical comparison is row03 P0-history/P0-boundary. The supplied-state reference under P1 history is a different comparison: it uses the same divergent incoming state as the failed adapter. It is not a canonical-history oracle.\n\n'
    report+='Recorded labels/predictions/CE match the trace and CSV associations. CE and probabilities were checked arithmetically from recorded logits using the locked numerical tolerance; these checks do not replace published results. No action-position grouping was used to infer tokens. Complete crosschecks are retained in arithmetic_and_identity_crosschecks.json.\n\n'
    report+='## Representation and coverage\n\n'
    for item in coverage:
        report+=f"- **{item['field']}**: {item['representation']}. {item['boundaries']}. {item.get('caveat',item.get('alternative',''))}\n"
    report+='\nPrefix JSON lacks direct commit snapshots because the observer writes the token trace before appending archival_commit to its returned summary (sealed research_programs_v4.py:390–396). Boundary current_cases summaries retain commit_pre/terminal/commit. A next incoming prefix/boundary snapshot is used as a returned-state observation only when its explicit case/history/token identity is unique and its incoming hash matches the previous outgoing hash. All such links and their coverage are recorded; this is not inference of raw tensor equality from summary equality.\n\n'
    report+='## Paired prefix timelines and first retained differences\n\n'
    for sid in selected:
        f=firsts.get(sid,{})
        p=f.get('first_prefix_program_difference');a=f.get('first_prefix_full_array_difference');b=f.get('final_P0_boundary_first_full_live_array_difference')
        report+=f"- **{sid}**: first prescribed action/block difference={p}; first retained full-array difference: token{a['token']}/decision{a['decision']}/{a['field']}, max|Δ|={a['comparison']['max_abs_difference']:.9g}, actual-array L2(Δ)={a['comparison']['L2_of_actual_difference']:.9g}. Final same-P0 boundary first retained live-array difference: decision{b['decision']}/{b['field']}, max|Δ|={b['max_abs_difference']:.9g}.\n"
    report+='\nThe differing prefix step executes A6/block6 for P0 versus REPEAT/block5 for P1. Later pointer/mask differences follow that prescribed substitution; they do not establish a defect. Prefix h0 and memory-read arrays and input summaries remain constant within each recorded token. Later tokens have differing incoming snapshots and preparation arrays. Full h, anchor, read and mean comparisons at every recorded decision are in array_differences.json/csv; all15-token paired timelines include logged output predictions/CE without turning them into retrospective gates.\n\n'
    report+='At the final same-P0 boundary, differing incoming observations precede the first retained live-array difference. Corresponding actions, executed blocks, P/D, cache versions, window/count metadata and consumed flags are compared separately from numerical states. The first retained observation in semantic reporting order is not a first causal error; fine-grained execution timestamps or causal interventions are unavailable.\n\n'
    report+='## Final-boundary state component summaries, all four cases\n\nSigned Δ means P1-history minus P0-history, in the *published summary statistic*. Neither Δnorm nor Δmax is a distance between the underlying tensors. No component is selected or ranked for causality.\n\n'
    for sid in selected:
        report+=f"### {sid}\n\n|Component|Incoming Δnorm|Incoming Δmax|Returned Δnorm|Returned Δmax|\n|---|---:|---:|---:|---:|\n"
        stage=boundary_components[sid]['summary_stages'];returned={r['field']:r for r in stage['committed_return']}
        for row in stage['incoming']:
            if row['representation']!='Retained summary' or 'summary_l2_delta' not in row:continue
            after=returned.get(row['field'],{})
            report+=f"|{row['field']}|{row['summary_l2_delta']:.9g}|{row['summary_max_abs_delta']:.9g}|{after.get('summary_l2_delta',float('nan')):.9g}|{after.get('summary_max_abs_delta',float('nan')):.9g}|\n"
    report+='\nAll stream, memory, GDN recurrence/buffer and attention cache summary fields are retained in component_summary_differences.json/csv for incoming, commit-pre and committed-return boundaries. Exact metadata is in recorded_metadata_comparisons.json and retained_protocol_consistency.json. If two summaries match, that is summary agreement only. GDN temporary states, complete memory slots, raw committed tensor values and separate rotary-position arrays are not available in the readable selected-case records.\n\n'
    report+='## Ledger arithmetic\n\nArchival16 + P0 prefixes1696 + P1 prefixes60 + current boundaries107 + supplied references100 =1979 completed tokens. Retained selected calls total15832 and commits1979. The67 P1 tokens comprise60 prefix tokens and7 boundary tokens. They execute block5 twice each:134 block5 calls,67 beyond one per token. Global block5 count2046 =1979+67; block6 count1912 =1979−67. REPEAT total67; RESTART0. This corroborates the logged program definitions, not an additional implementation audit. No future records/calls or final-joint bootstrap outputs exist.\n\n'
    report+='## What the evidence cannot determine\n\nThe observed state fingerprints, summaries and array differences do not establish which component represents parity, whether information was irreversibly lost, or which component caused the row03 error. We cannot compute raw GDN/memory/cache distances from norm summaries, inspect ephemeral GDN state, establish equivalent future behavior, or predict the unexecuted row03 P1→P1 result. Serialized state files for the three controls exist but were not deserialized; row03 has no saved bundle. A prior runtime full-state comparison is not a new offline tensor comparison. No population estimates, confidence intervals, calibrated semantic probe or component-selection rule is inferred from these four ordered cases.\n\n'
    report+='## Assessment: program-dependent readability\n\nThe retained case supports asking a separately preregistered question about whether a P1-generated committed state may yield different task-readable behavior under different boundary programs. It does not answer that question: row03’s P1→P1 boundary is unobserved, and present-output correctness elsewhere does not prove recoverable future information. The three controls show that switching failure is not universal in the observed cases. Readability mismatch, accumulated state error and other coupled effects remain unresolved. No new runner, component chosen by largest summary difference, or follow-up execution is proposed here. P1 is not reinterpreted as passing v4.\n\n'
    after={name:dict(**info,sha256_after=sha(INPUT/name),equal=sha(INPUT/name)==info['sha256']) for name,info in before['evidence_files'].items()}
    diff_sha=hashlib.sha256(subprocess.check_output(['git','diff','--binary'])).hexdigest()
    integrity=dict(evidence_files=after,all_evidence_hashes_equal=all(v['equal'] for v in after.values()),
        tracked_diff_sha256_after=diff_sha,tracked_diff_unchanged=diff_sha==before['repository']['tracked_diff_sha256'],
        historical_v4_integrity=verification,
        broader_inventory_reverification=False,checkpoint_hash_reverification=False,
        historical_integrity_exception_status='Carried unchanged from v4 findings; no new cache/model imports',
        no_model_import=True,no_checkpoint_deserialization=True,no_execution=True,no_tests=True,
        forbidden_modules_loaded=[name for name in sys.modules if name=='torch' or name.startswith('src.') or name.startswith('experiments.')],
        no_new_noise=True,no_bootstrap=True,no_commit=True,new_files_scope=str(OUT.relative_to(ROOT)))
    write_json('integrity_after.json',integrity)
    report+=f"## Integrity and execution boundary\n\nAll{len(after)} v4 evidence files read had identical before/after SHA-256 hashes. Original tracked-diff identity also matches. V4’s broader finding that617 protected identities/checkpoints and historical bytecode exception were unchanged is carried forward, **not newly reverified** by this analysis. No checkpoints or state bundles were deserialized. All new files are confined to {OUT.relative_to(ROOT)}. No model import/construction, forward/backward, test, optimizer, training, selector fitting, noise, bootstrap, production edit or commit occurred.\n"
    with (OUT/'report.md').open('x') as handle:handle.write(report)
    write_json('analysis_summary.json',dict(selected_ids=selected,four_cells=four_cells,
        first_retained_differences=firsts,ledger=reconciled,coverage_ambiguities=ambiguities,
        missing=missing,arithmetic_checks=consistency['all_CE_checks_within_locked_scalar_tolerance'],
        all_evidence_hashes_equal=integrity['all_evidence_hashes_equal'],
        interpretation='Recorded current/switch failure preserved; readability question motivated but unanswered; no causal attribution'))
    print(json.dumps(dict(selected_ids=selected,cells=[{k:v for k,v in c.items() if k in ['sequence_id','history_program','boundary_program','status','label','prediction','CE']} for c in four_cells],
        ledger=reconciled,ambiguities=ambiguities,missing=missing,
        arithmetic_consistent=consistency['all_CE_checks_within_locked_scalar_tolerance'],
        evidence_hashes_equal=integrity['all_evidence_hashes_equal']),indent=2))


if __name__=='__main__':
    execute()
