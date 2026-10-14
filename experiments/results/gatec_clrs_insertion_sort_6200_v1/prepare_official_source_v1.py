"""Fetch only unchanged official sampler import-closure source, never datasets."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

OUT = Path(__file__).resolve().parent
REV = '6d5728b9efadf8648413c8c473929a733caacced'
PATHS = ['LICENSE', 'clrs/__init__.py'] + [
    'clrs/_src/' + n + '.py' for n in ('samplers', 'probing', 'specs', 'evaluation')
] + ['clrs/_src/algorithms/__init__.py'] + [
    'clrs/_src/algorithms/' + n + '.py' for n in (
        'divide_and_conquer', 'dynamic_programming', 'geometry', 'graphs',
        'greedy', 'searching', 'sorting', 'strings')
]

def main():
    destination = OUT / 'official_source'
    if destination.exists():
        raise SystemExit('STOP: official source destination exists')
    destination.mkdir()
    records = []
    for name in PATHS:
        url = 'https://raw.githubusercontent.com/google-deepmind/clrs/' + REV + '/' + name
        payload = urllib.request.urlopen(url, timeout=40).read()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(payload)
        imports = []
        if name.endswith('.py'):
            for node in ast.walk(ast.parse(payload)):
                if isinstance(node, ast.Import):
                    imports.extend(x.name for x in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module)
        records.append(dict(path=name, url=url, size=len(payload),
                            sha256=hashlib.sha256(payload).hexdigest(), imports=imports))
    (OUT / 'official_source_manifest_v1.json').write_text(json.dumps(
        dict(revision=REV, files=records, source_edits=False,
             import_strategy='Namespace clrs and clrs._src only; unchanged official algorithms package, sampler, probing and evaluation. Top-level convenience __init__ not executed; avoids models/TFDS.'),
        indent=2) + '\n')
    print('Official source files:', len(records), 'bytes:', sum(x['size'] for x in records))

if __name__ == '__main__':
    main()
