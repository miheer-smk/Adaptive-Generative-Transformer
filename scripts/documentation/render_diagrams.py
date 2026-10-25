#!/usr/bin/env python3
"""Render editable source-grounded graphs as Mermaid, SVG and PNG; no model code."""
from pathlib import Path
import json,hashlib,math
ROOT=Path(__file__).resolve().parents[2]
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'agt-diagrams-v1'})
    graphs=json.loads((ROOT/'docs/figure_sources/diagrams.json').read_text());manifest=[];gallery=['# Editable Mermaid diagrams','',
      'The SVG/PNG fallbacks and these Mermaid sources share one editable graph definition. Static images use the coordinates in diagrams.json; Mermaid chooses its own layout. Arrows do not assert scientific validation.','']
    palette={'blue':('#EAF3F8','#0072B2'),'orange':('#FFF1E9','#B14A18'),'gray':('#F0F2F4','#6C7884')}
    for name,g in graphs.items():
        w,h=g['size'];fig,ax=plt.subplots(figsize=(w,h));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.89);ax.set_xlim(0,w);ax.set_ylim(0,h-.7);ax.axis('off')
        nodes={n['id']:n for n in g['nodes']};mmd=['flowchart LR']
        for n in g['nodes']:
            face,border=palette[n['color']];box=FancyBboxPatch((n['x']-n['w']/2,n['y']-n['h']/2),n['w'],n['h'],boxstyle='round,pad=0.10,rounding_size=.09',facecolor=face,edgecolor=border,linewidth=1.2,zorder=3);ax.add_patch(box)
            ax.text(n['x'],n['y'],n['text'],ha='center',va='center',fontsize=9.4 if name!='token_execution' else 9.1,color='#183247',linespacing=1.45,zorder=4)
            label=n['text'].replace('\n','<br/>').replace('"',"'");mmd.append(f'    {n["id"]}["{label}"]')
        for e in g['edges']:
            a=nodes[e['source']];b=nodes[e['target']];dx=b['x']-a['x'];dy=b['y']-a['y']
            def anchor(n,xx,yy):
                scale=1/max(abs(xx)/(n['w']/2+.13),abs(yy)/(n['h']/2+.13));return (n['x']+xx*scale,n['y']+yy*scale)
            start=anchor(a,dx,dy);end=anchor(b,-dx,-dy)
            arrow=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,linewidth=1.0,color='#617381',linestyle='--' if e['style']=='dashed' else '-',connectionstyle=f'arc3,rad={e["rad"]}',zorder=1);ax.add_patch(arrow)
            if e['label']:
                # A light text box keeps edge labels legible over crossing lines.
                ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.12,e['label'],ha='center',va='center',fontsize=7.2,color='#445B6C',bbox=dict(facecolor='white',edgecolor='none',pad=1.5),zorder=5)
            arrowtext='-.->' if e['style']=='dashed' else '-->'
            mmd.append(f'    {e["source"]} {arrowtext}'+(f'|"{e["label"]}"|' if e['label'] else '')+f' {e["target"]}')
        fig.suptitle(g['title'],fontsize=15,fontweight='bold',color='#172B3A',y=.98);fig.text(.5,.915,g['scope'],ha='center',fontsize=9.5,color='#536574')
        source='\n'.join(mmd)+'\n';(ROOT/f'docs/figure_sources/{name}.mmd').write_text(source)
        for ext in ['svg','png']:fig.savefig(ROOT/f'docs/figures/{name}.{ext}',dpi=150,metadata={'Date':None} if ext=='svg' else {},facecolor='white')
        svg=ROOT/f'docs/figures/{name}.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig);gallery += [f'## {g["title"]}','',g['scope'],'','```mermaid',source.rstrip(),'```','']
        manifest.append(dict(name=name,caption=g['title'],scope=g['scope'],editable_source=f'docs/figure_sources/{name}.mmd',graph_definition='docs/figure_sources/diagrams.json',outputs=[f'docs/figures/{name}.{e}' for e in ['svg','png']],source_map='docs/architecture.md' if name!='evidence_map' else 'docs/experiment_index.md',command='python scripts/documentation/render_diagrams.py',rendering='Matplotlib fallback; Mermaid source generated from identical graph. Mermaid engine validation is recorded separately.'))
    (ROOT/'docs/figure_sources/mermaid_diagrams.md').write_text('\n'.join(gallery).rstrip()+'\n')
    (ROOT/'docs/figure_sources/diagram_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Rendered {len(graphs)} editable diagrams as Mermaid + SVG + PNG.')
if __name__=='__main__':main()
