#!/usr/bin/env python3
"""Build navigation from curated records and an audited availability snapshot.

Standard library only. Never imports model code, loads checkpoints or runs tests.
"""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
REPO='https://github.com/miheer-smk/Adaptive-Generative-Transformer'
def link(path,commit,label=None):
    from urllib.parse import quote
    return f"[{label or Path(path).name}]({REPO}/blob/{commit}/{quote(path,safe='/')})"
def main():
    catalog=json.loads((ROOT/'docs/figure_sources/milestones.json').read_text())
    available=json.loads((ROOT/'docs/maintenance/evidence_availability.json').read_text())
    branches=json.loads((ROOT/'docs/maintenance/repository_state_20261025.json').read_text())['research_branches']
    records=[]
    for row in catalog['milestones']:
        r=dict(row)
        paths=[r['report'],r['protocol'],r['reproduction_entry'],*r['result_data'],*r['checkpoints']]
        r['artifact_records']={p:available['files'][r['evidence_commit']+':'+p] for p in paths if p}
        r['known_publication_branches']=[n for n,s in branches.items() if s==r['evidence_commit']]
        records.append(r)
    result=dict(as_of=catalog['as_of'],canonical_evidence_snapshot=catalog['canonical_evidence_snapshot'],revision_semantics='Evidence-link commits are immutable publication snapshots, not assertions that execution used a clean checkout of that commit.',milestones=records)
    (ROOT/'docs/evidence_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Experiment and evidence index','',
        'Audited **2026-10-25**. This index preserves recorded classifications; editorial summaries are explicitly separate. The latest completed milestone is the feature-complement pilot. No executed successor or newly authorized next experiment was found.','',
        'The remote default branch is an older history snapshot. The evidence links below are commit-pinned. A publication commit identifies accessible evidence, not necessarily the original execution HEAD: several early runs used an uncommitted, fingerprinted source diff. See [repository state](maintenance/repository_state_20261025.json), [machine-readable inventory](evidence_inventory.json), [reproduction boundaries](reproducibility.md), and [known discrepancies](limitations.md#evidence-conflicts-and-unavailable-material).','',
        '## Milestone map','',
        '|Milestone|Scientific question|Recorded status or report verdict|Evidence|Scope limit|','|---|---|---|---|---|']
    for r in records:
        lines.append(f"|[{r['name']}](#{r['id']})|{r['question']}|{r['recorded_status']}|{link(r['report'],r['evidence_commit'],'Report')}|{r['limitations']}|")
    lines+=['','## Full provenance and reproduction records','',
        'An entry point is a source pointer, **not a claim that running it in place is safe or portable**. Most historical runners deliberately refuse an existing output directory and some assume DGX-specific paths. No experiment runner or model test was executed to prepare this index. Availability refers to the audited Git tree; LFS payload reachability is qualified separately.','']
    for r in records:
        c=r['evidence_commit'];lines += [f"<a id=\"{r['id']}\"></a>",f"### {r['name']}",'',f"**Question:** {r['question']}",'',f"**Recorded status/verdict:** `{r['recorded_status']}`. **Evidence type:** {r['kind']}.",'',f"**Observed:** {r['editorial_summary']}",'',f"**Configuration/data:** {r['configuration']}",'',f"**Limit:** {r['limitations']}",'',f"**Report:** {link(r['report'],c,r['report'])}."]
        lines.append(f"**Protocol/configuration:** {link(r['protocol'],c,r['protocol'])}." if r['protocol'] else '**Protocol/configuration:** no separate locked protocol located for this record; use the report and linked parent specifications.')
        lines.append('**Numerical records:** '+', '.join(link(p,c,p) for p in r['result_data'])+'.' if r['result_data'] else '**Numerical records:** report tables and linked historical artifacts; no additional aggregate JSON is asserted here.')
        lines.append(f"**Source entry:** {link(r['reproduction_entry'],c,r['reproduction_entry'])}." if r['reproduction_entry'] else '**Source entry:** inspect the report artifact list; no single portable reproduction entry is asserted.')
        lines.append(f"**Evidence revision:** [`{c[:12]}`]({REPO}/commit/{c}). {r['evidence_revision_role']}.")
        if r['known_publication_branches']:lines.append('**Known branch:** '+', '.join('`'+n+'`' for n in r['known_publication_branches'])+'.')
        if r['checkpoints']:
            lines += ['','|Checkpoint|SHA-256|Availability|','|---|---|---|']
            for p in r['checkpoints']:
                a=r['artifact_records'][p];lines.append(f"|{link(p,c,p)}|`{a['local_sha256']}`|{a['availability']}|")
        lines += ['',r['reproduction_requirements'],'']
    (ROOT/'docs/experiment_index.md').write_text('\n'.join(lines).rstrip()+'\n')
    print(f'Indexed {len(records)} milestones; no model or checkpoint deserialization.')
if __name__=='__main__':main()
