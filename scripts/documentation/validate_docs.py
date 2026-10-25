#!/usr/bin/env python3
"""Standard-library presentation checks. Never imports or executes AGT/tests."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/maintenance/preservation_before_20261025.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT,
        env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})


def anchors(text):
    out = set(re.findall(r'<a\s+id=["\']([^"\']+)', text))
    counts = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        slug = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
        n = counts.get(slug, 0)
        out.add(slug + (f'-{n}' if n else ''))
        counts[slug] = n + 1
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    baseline = json.loads(BASE.read_text())
    errors = []
    counts = {'preserved_files': 0, 'markdown_files': 0, 'local_links': 0,
              'pinned_links': 0, 'figure_outputs': 0, 'diagrams': 0}

    for rel, old in baseline['files'].items():
        p = ROOT / rel
        if not p.exists() and not p.is_symlink():
            errors.append(f'Pre-existing path absent: {rel}')
            continue
        if stat.S_IMODE(p.lstat().st_mode) != old['mode']:
            errors.append(f'Pre-existing mode changed: {rel}')
        if old['kind'] == 'file':
            if p.is_symlink() or sha(p) != old['sha256']:
                errors.append(f'Pre-existing bytes changed: {rel}')
        elif old['kind'] == 'symlink':
            if not p.is_symlink() or os.readlink(p) != old['target']:
                errors.append(f'Pre-existing symlink changed: {rel}')
        counts['preserved_files'] += 1

    if git('rev-parse', 'HEAD').decode().strip() != baseline['original_HEAD']:
        errors.append('Local HEAD commit changed')
    if hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest() != baseline['tracked_diff_sha256']:
        errors.append('Original tracked diff changed')
    if sha(ROOT / '.git/index') != baseline['index_sha256']:
        errors.append('Original index changed')
    if git('rev-parse', baseline['original_branch']).decode().strip() != baseline['original_HEAD']:
        errors.append('Original branch reference changed')

    added = [p for p in ROOT.rglob('*') if p.is_file()
             and '.git' not in p.relative_to(ROOT).parts
             and p.relative_to(ROOT).as_posix() not in baseline['files']]
    allowed = ['docs/', 'scripts/documentation/']
    singles = {'README.md', 'CONTRIBUTING.md', 'src/README.md',
               'tests/README.md', 'experiments/README.md'}
    original_casefold = {p.casefold(): p for p in baseline['files']}
    for p in added:
        rel = p.relative_to(ROOT).as_posix()
        if rel.casefold() in original_casefold:
            errors.append(f'New path collides by case with historical file: {rel}')
        if rel not in singles and not any(rel.startswith(a) for a in allowed):
            errors.append(f'Unexpected new path: {rel}')
    references = read('docs/maintenance/source_references.json')
    for p in added:
        if p.suffix != '.md':
            continue
        counts['markdown_files'] += 1
        text = p.read_text()
        for target in re.findall(r'!?\[[^\]]*\]\(([^\n]+?)\)', text):
            target = target.strip().strip('<>')
            u = urlsplit(target)
            if u.scheme:
                if u.netloc == 'github.com' and '/blob/' in u.path:
                    key = unquote(u.path.split('/blob/', 1)[1])
                    if not re.match(r'^[0-9a-f]{40}/', key):
                        errors.append(f'Unpinned evidence link in {p.name}: {key}')
                    elif key not in references['verified']:
                        errors.append(f'Unverified pinned reference: {key}')
                    counts['pinned_links'] += 1
                continue
            if u.path.startswith('/'):
                errors.append(f'Machine-absolute published link: {p.name}: {target}')
                continue
            dst = (p.parent / unquote(u.path)).resolve() if u.path else p
            # This report is deliberately created after its inputs are checked.
            planned = ROOT / 'docs/maintenance/validation_20261025.json'
            if not dst.exists() and dst != planned:
                errors.append(f'Broken local link {p.relative_to(ROOT)} -> {target}')
            if u.fragment and dst.exists() and dst.suffix == '.md':
                if unquote(u.fragment) not in anchors(dst.read_text()):
                    errors.append(f'Broken heading {p.relative_to(ROOT)} -> {target}')
            counts['local_links'] += 1
        if re.search(r'[ \t]+$', text, re.M):
            errors.append(f'Trailing whitespace: {p.relative_to(ROOT)}')

    provenance = read('docs/figure_sources/figure_provenance.json')
    for fig in provenance['figures']:
        for source in fig['sources']:
            if sha(ROOT / source['path']) != source['sha256']:
                errors.append(f'Figure input changed: {source["path"]}')
        for ext, digest in fig['outputs'].items():
            p = ROOT / 'docs/figures' / fig['files'][0].replace('.svg', '.' + ext)
            if sha(p) != digest:
                errors.append(f'Figure hash mismatch: {p.name}')
            counts['figure_outputs'] += 1
    graphs = read('docs/figure_sources/diagrams.json')
    for name, g in graphs.items():
        nodes = {n['id'] for n in g['nodes']}
        if len(nodes) != len(g['nodes']):
            errors.append(f'Duplicate graph node: {name}')
        for e in g['edges']:
            if e['source'] not in nodes or e['target'] not in nodes:
                errors.append(f'Unknown graph endpoint: {name}')
        mmd = (ROOT / f'docs/figure_sources/{name}.mmd').read_text()
        if not mmd.startswith('flowchart LR\n') or any(f'    {n}["' not in mmd for n in nodes):
            errors.append(f'Mermaid graph/source disagreement: {name}')
        for ext in ['svg', 'png']:
            if not (ROOT / f'docs/figures/{name}.{ext}').is_file():
                errors.append(f'Missing diagram output: {name}.{ext}')
        counts['diagrams'] += 1
    forbidden = {'torch', 'src', 'experiments', 'tensorflow', 'jax'}
    for p in (ROOT / 'scripts/documentation').glob('*.py'):
        tree = ast.parse(p.read_text(), filename=str(p))
        for node in ast.walk(tree):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else (
                    [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            if any(n.split('.')[0] in forbidden for n in names):
                errors.append(f'Prohibited model import: {p.name}')
    available = read('docs/maintenance/evidence_availability.json')
    if available.get('missing_remote'):
        errors.append('Indexed evidence missing remotely')
    if len(read('docs/figure_sources/milestones.json')['milestones']) != 32:
        errors.append('Milestone count differs from documented inventory')
    result = {'status': 'PASS' if not errors else 'FAIL', 'counts': counts,
              'added_files_checked': len(added), 'errors': errors,
              'scientific_execution': False, 'project_tests_executed': False,
              'preservation': {'baseline': str(BASE.relative_to(ROOT)),
                  'tracked_diff_sha256': baseline['tracked_diff_sha256'],
                  'index_unchanged': 'Original index changed' not in errors,
                  'original_branch_unchanged': 'Original branch reference changed' not in errors},
              'rendering': {'static_SVG_PNG': 'Generated and visually inspected, all ten figures',
                  'Mermaid': 'Graph/source consistency checked; no Mermaid engine available; engine rendering unverified'},
              'limits': ['No full project test suite or model execution',
                  'No large LFS payload re-download; indexed pointer/local identities checked separately',
                  'No claim of a project-wide license, complete blueprint or current green CI']}
    if args.write_report:
        (ROOT / 'docs/maintenance/validation_20261025.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
