#!/usr/bin/env python3
"""Render retained aggregate results only. No torch/model/runner imports or fits."""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'experiments/results'
SOURCES={
 'original':('gatec_clrs_insertion_sort_6200_v1/training_results_v1/final_results_v1.json','386ea1be31abe05d1ccf31f59511c1fd1d470aed'),
 'fit':('gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/metrics_v1.json','94a7b19079dad3fd21afc801f62d65ed9101b613'),
 'refit':('gatec_clrs_bilinear_refit_6300_v1/metrics_v1.json','97f3b78a17d4060d2b0fef8b52c606596de6ca5e'),
 'feature':('gatec_clrs_feature_complement_pilot_6400_v1/evaluation_summaries_v1.json','ea183970feea1f49677f186e26065ce4e333f709'),
 'contrast':('gatec_clrs_feature_complement_pilot_6400_v1/final_results_v1.json','ea183970feea1f49677f186e26065ce4e333f709'),
 'geometry':('gatec_clrs_feature_complement_pilot_6400_v1/mechanistic_summary_v1.json','ea183970feea1f49677f186e26065ce4e333f709'),
 'receiver':('gatec_receiver_anchored_compatibility_6000_v1/final_analysis.json','1dd215e24407b8e497abd7e4c92de5702fc25821'),
 'gradient':('gatec_paired_gradient_average_4900_v1/results.json','1dd215e24407b8e497abd7e4c92de5702fc25821'),
 'mhc':('gatec_clrs_mhc_capture_6200_v1/blockwise_summary_v1.csv','d9d5eb0061b860b796f64fbc559420caff863509')}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def inputs():
    expected=json.loads((ROOT/'docs/figure_sources/result_source_hashes.json').read_text());values={};records={}
    for key,(name,commit) in SOURCES.items():
        path=P/name
        if not path.is_file():raise SystemExit(f'Missing retained artifact: {path.relative_to(ROOT)}. Obtain it at commit {commit}; model execution is not a fallback.')
        with path.open('rb') as f:head=f.read(90)
        if head.startswith(b'version https://git-lfs.github.com/spec/v1'):raise SystemExit(f'LFS pointer instead of payload: {path.relative_to(ROOT)}. Materialize the documented existing LFS object first.')
        digest=sha(path)
        if digest!=expected[key]['sha256']:raise SystemExit(f'Source hash mismatch: {path.relative_to(ROOT)}')
        records[key]=dict(path=path.relative_to(ROOT).as_posix(),commit=commit,sha256=digest)
        values[key]=list(csv.DictReader(path.open())) if path.suffix=='.csv' else json.loads(path.read_text())
    return values,records
def data(v):
    # Explicit fractions from existing aggregates; no uncertainty is recomputed.
    clrs={}
    for split in ['development','size_transfer']:
        original=v['original']['metrics'][split]
        clrs[split]={
          'Original native':dict(pointer=original['pointer_accuracy'],exact=original['exact_array_accuracy']),
          'Frozen fit (uncertified)':dict(pointer=v['fit'][split]['fitted_FP32']['predecessor_accuracy'],exact=v['fit'][split]['fitted_FP32']['exact_array_accuracy']),
          'Certified refit':dict(pointer=v['refit'][split]['fitted_FP32']['predecessor_accuracy'],exact=v['refit'][split]['fitted_FP32']['exact_array_accuracy']),
          'Feature control':dict(pointer=v['feature'][split]['C']['native']['predecessor_accuracy'],exact=v['feature'][split]['C']['native']['exact_array_accuracy']),
          'Feature treatment':dict(pointer=v['feature'][split]['T']['native']['predecessor_accuracy'],exact=v['feature'][split]['T']['native']['exact_array_accuracy'])}
    contrasts={k:dict(point=x['point_estimate'],lower=x['adjusted_one_sided_lower_bound']) for k,x in v['contrast']['contrasts'].items()}
    receiver={a:v['receiver']['robust_patterns'][a]['1111']/sum(v['receiver']['robust_patterns'][a].values()) for a in ['C','T']}
    gradient={s:{a:v['gradient']['pooled_stochastic'][a][s]['accuracy'] for a in ['C','T']} for s in ['val','ood_test']}
    for s in gradient:gradient[s]['reference']=v['gradient']['reference'][s]['accuracy']
    return dict(clrs=clrs,feature_contrasts=contrasts,receiver_all_cells_robust=receiver,receiver_ablation=v['receiver']['ablation']['point'],receiver_delta=v['receiver']['primary_delta'],receiver_interval=v['receiver']['primary_interval'],gradient=gradient,geometry=v['geometry'],mhc=v['mhc'])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-data',action='store_true');args=parser.parse_args()
    v,records=inputs();d=data(v);dest=ROOT/'docs/figure_sources/figure_data.json'
    if args.check_data:
        assert json.loads(dest.read_text())==d,'Committed plotted data differs from retained sources'
        print('All plotted data match retained artifacts; no model imported.');return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#263445','text.color':'#172B3A','axes.edgecolor':'#A3AFB8','svg.hashsalt':'agt-documentation-v1','figure.facecolor':'white','savefig.facecolor':'white'})
    output=ROOT/'docs/figures';output.mkdir(parents=True,exist_ok=True);manifest=[]
    blue='#0072B2';orange='#D55E00';gray='#6C7884';purple='#7356A6';light='#95A8B8'
    def finish(fig,name,caption,keys,transforms,limits):
        fig.savefig(output/(name+'.svg'),bbox_inches='tight',metadata={'Date':None,'Description':caption})
        fig.savefig(output/(name+'.png'),dpi=150,bbox_inches='tight',metadata={'Description':caption})
        svg = output/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
        manifest.append(dict(files=[name+'.svg',name+'.png'],caption=caption,sources=[records[k] for k in keys],transformations=transforms,command='python scripts/documentation/render_figures.py',limits=limits,generator='Matplotlib '+matplotlib.__version__,outputs={ext:sha(output/(name+'.'+ext)) for ext in ['svg','png']}))
    # 1: All bars use the full percentage range, including the zero exact scores.
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.1),layout='constrained');labels=list(d['clrs']['development']);colors=[gray,light,purple,blue,orange]
    for ax,metric,title in zip(axs,['pointer','exact'],['Predecessor accuracy','Exact-array accuracy']):
        for j,label in enumerate(labels):
            vals=[100*d['clrs'][s][label][metric] for s in ['development','size_transfer']];xs=[x+(j-2)*.145 for x in [0,1]]
            bars=ax.bar(xs,vals,width=.135,color=colors[j],label=label)
            ax.bar_label(bars,labels=[f'{x:.1f}' for x in vals],padding=3,fontsize=8,rotation=90 if metric=='pointer' else 0)
        ax.set_xticks([0,1],['development\nn=16; 32 arrays','size_transfer\nn=64; 32 arrays']);ax.set_ylim(0,105);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Accuracy (%)');ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.hlines([95,90] if metric=='pointer' else [80,50],[-.40,.60],[.40,1.40],colors='#303A43',linestyles='dashed',linewidth=1.3)
    handles,labels2=axs[0].get_legend_handles_labels();fig.legend(handles,labels2,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('CLRS insertion sort: no variant meets the declared competence gates',fontsize=14)
    finish(fig,'clrs_outcomes','Native training, two separately sealed frozen-core readouts and the paired feature-complement arms on reused panels. Every exact-array score is zero; dashed segments show the original competence thresholds.',['original','fit','refit','feature'],'Fractions multiplied by100; no error bars or newly estimated uncertainty. Each panel uses32 complete arrays (512 or2048 pointers).','Variants are different studies, not independent replicates. Fits use frozen original states; feature arms train fresh cores. No official CLRS GNN comparison.')
    # 2: A lower bound is not presented as a symmetric or two-sided interval.
    fig,ax=plt.subplots(figsize=(9,3.7),layout='constrained')
    for y,(name,x) in enumerate(d['feature_contrasts'].items()):
        point=x['point']*100;lower=x['lower']*100;ax.plot([lower,point],[y,y],color=gray,lw=1.5);ax.scatter([point],[y],s=70,c=orange,label='Point estimate' if y==0 else None,zorder=3);ax.scatter([lower],[y],s=60,marker='|',c='#172B3A',label='Adjusted one-sided lower bound' if y==0 else None,zorder=4)
        ax.text(point+.22,y,f'point {point:.2f}; lower {lower:.2f}',va='center',fontsize=10)
    ax.set_yticks([0,1],['Treatment − matched control','Treatment − own query-independent component']);ax.invert_yaxis();ax.set_xlim(-.3,11.3);ax.axvline(0,color=gray,lw=.8);ax.set_xlabel('Development predecessor accuracy difference (percentage points)');ax.set_title('Feature complement: partial gain, no resolved query-conditioned advantage');ax.legend(loc='lower right',frameon=False,fontsize=9);ax.set_ylim(1.6,-.6)
    finish(fig,'feature_contrasts','Two locked development contrasts: point estimates and retained adjusted one-sided lower bounds. The connecting segment is not a two-sided confidence interval.',['contrast'],'Existing32-array,20000-replicate seed6400 bootstrap results multiplied by100; no resampling here.','One paired seed and previously examined development panel. A degenerate zero bound does not prove universal equivalence. Both competence panels failed.')
    # 3: Why saturation did not satisfy the receiver-anchored study.
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');vals=[100*d['receiver_all_cells_robust'][a] for a in ['C','T']]
    bars=axs[0].bar(['Matched native control','Receiver-anchored treatment'],vals,color=[blue,orange]);axs[0].bar_label(bars,fmt='%.0f%%',padding=3);axs[0].set_ylim(0,110);axs[0].set_ylabel('Fully robust cases (%)');axs[0].set_title('All four boundary cells fully robust')
    bars=axs[1].bar(['C/P0','C/P1','T/P0','T/P1'],[100*x for x in d['receiver_ablation']],color=[blue,blue,orange,orange]);axs[1].bar_label(bars,fmt='%.0f',padding=3);axs[1].axhline(5,color=gray,linestyle='--',label='Required point decrease: 5 pp');axs[1].set_ylim(0,10);axs[1].set_ylabel('Ablation accuracy decrease (pp)');axs[1].set_title('Distinguishing-operation ablations');axs[1].legend(frameon=False,loc='upper right',fontsize=8)
    fig.suptitle('Receiver-anchored parity study: COMPLETED_NON_SUCCESS',fontsize=14)
    finish(fig,'receiver_outcome','Both arms have512/512 cases fully robust across all four cells, but primary delta=0 with retained95% interval[0,0]; all four ablation decreases and adjusted lower bounds are0.',['receiver'],'Robust-pattern1111 count divided by512; retained ablation fractions multiplied by100. No new confidence estimates.','Parity saturated both arms. Neither a compatibility-training benefit nor task-relevant heterogeneous operations were established; one initialization, not seed replication.')
    # 4: Same declared input panels; four tapes are repeated observations.
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.4),layout='constrained')
    for ax,s,title in zip(axs,['val','ood_test'],['val: 32 length16 sequences','ood_test: 64 sequences, lengths17–32']):
        vals=[100*d['gradient'][s][a] for a in ['reference','C','T']];bars=ax.bar(['Canonical\nreference','Duplicate-noise\ncontrol','Independent-noise\ntreatment'],vals,color=[gray,blue,orange]);ax.bar_label(bars,fmt='%.1f',padding=3);ax.axhline(50,color=light,linestyle=':',linewidth=1);ax.set_ylim(0,110);ax.set_ylabel('Parity accuracy (%)');ax.set_title(title)
    fig.suptitle('Two-gradient averaging did not repair task-loss routing discovery',fontsize=14)
    finish(fig,'gradient_averaging','Declared final paired gradient-averaging evaluation: canonical reference versus duplicate-noise and independent-noise controller arms.',['gradient'],'C/T pooled accuracy across four locked tapes; reference is deterministic on the same unique inputs. No error bars.','One initialization.128/256 tape-sequence observations correspond to32/64 unique sequences; tapes are not independent datasets. No useful-policy repair under the locked rule.')
    # 5: Actual retained core-side mixer tensors, never inferred via inversion.
    fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for ai,role,title in [(0,'candidate','Candidate positions'),(1,'query','Query positions')]:
        rows=sorted([r for r in d['mhc'] if r['positions']==role],key=lambda r:int(r['block']));xs=list(range(8));inp=[float(r['input_centered_energy']) for r in rows];out=[float(r['output_centered_energy']) for r in rows]
        assert min(inp+out)>0;axs[ai].plot(xs,inp,'o-',color=blue,label='Mixer input');axs[ai].plot(xs,out,'s--',color=orange,label='Mixer output');axs[ai].set_yscale('log');axs[ai].set_xticks(xs);axs[ai].set_xlabel('Bank block index');axs[ai].set_ylabel('Absolute centered energy (log scale)');axs[ai].set_title(title);axs[ai].legend(frameon=False,fontsize=8)
        offset=-.17 if ai==0 else .17;axs[2].bar([x+offset for x in xs],[float(r['energy_ratio']) for r in rows],width=.30,color=blue if ai==0 else orange,label=role)
    axs[2].axhline(1,color=gray,ls=':',lw=1);axs[2].set_ylim(0,1.2);axs[2].set_xticks(range(8));axs[2].set_xlabel('Bank block index');axs[2].set_ylabel('Output/input centered energy');axs[2].set_title('Observed mixer energy transfer');axs[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Frozen mHC capture: one training array, one failed final checkpoint',fontsize=14)
    finish(fig,'mhc_observed_transfer','Actual captured mixer inputs/outputs for16 candidate and16 query positions of official/train/row0000. Absolute energies use a labeled log scale; ratios start at zero.',['mhc'],'Read existing CSV input/output centered energies and ratios; no activation reconstruction or new model calls.','Attention-mixer attenuation is directly observed, but task relevance and historical causation are not established. The surrounding residual path can preserve distinctions.')
    # 6: Geometry can improve without a competent native task solution.
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.3),layout='constrained')
    for ai,(metric,label) in enumerate([('mean_output_centered_energy','Mean committed centered energy'),('mean_output_angular_dispersion','Mean committed angular dispersion')]):
        for arm,color,offset,marker in [('C',blue,-.07,'o'),('T',orange,.07,'s')]:
            vals=[d['geometry'][arm]['development'][role]['terminal'][metric] for role in ['candidates','queries']];assert min(vals)>0;axs[ai].scatter(vals,[0+offset,1+offset],color=color,marker=marker,s=65,label=arm)
        axs[ai].set_xscale('log');axs[ai].set_yticks([0,1],['Candidates','Queries']);axs[ai].invert_yaxis();axs[ai].set_xlabel(label+' (log scale)');axs[ai].grid(axis='x',alpha=.15);axs[ai].legend(title='Arm',frameon=False)
    fig.suptitle('Feature complement: more retained variation did not restore competence',fontsize=14)
    finish(fig,'feature_terminal_geometry','Development means over32 arrays, separately for candidate/query positions in the final trained control and treatment.',['geometry'],'Existing terminal summary means; dot plots use log axes to show several orders of magnitude. No error bars or new statistics.','Arms have different learned parameters and trajectories. Geometry is descriptive, not a same-weight component intervention, task-relevance proof or universal collapse criterion.')
    dest.write_text(json.dumps(d,indent=2)+'\n')
    (ROOT/'docs/figure_sources/figure_provenance.json').write_text(json.dumps(dict(schema_version=1,figures=manifest,scientific_execution=False),indent=2)+'\n')
    print(f'Rendered {len(manifest)} result figures as SVG + PNG; no model imported.')
if __name__=='__main__':main()
