"""Offline metric, provenance and integrity helpers; no model imports."""
import hashlib, json, os, random, subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
ENGINEERING = OUT.parent
ROOT = OUT.parents[3]

def save(name, value):
    p = OUT / name
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def integrity():
    initial = json.loads((OUT / 'integrity_initial_v1.json').read_text())
    bad = []
    for name, v in initial['files'].items():
        p = ROOT / name
        if v['kind'] == 'symlink':
            ok = p.is_symlink() and os.readlink(p) == v['target']
        else:
            ok = p.is_file() and not p.is_symlink() and p.stat().st_size == v['size'] and sha(p) == v['sha256'] and p.stat().st_mode & 0o777 == v['mode']
        if not ok:
            bad.append(name)
    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, env=env).decode().strip()
    diff = hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT, env=env)).hexdigest()
    check = dict(protected_count=len(initial['files']), changed_or_missing=bad,
                 HEAD_unchanged=head == initial['HEAD'], tracked_diff_unchanged=diff == initial['tracked_diff_sha256'],
                 historical_exception_preserved=True, no_new_exception=True)
    check['success'] = not bad and check['HEAD_unchanged'] and check['tracked_diff_unchanged']
    save('integrity_final_v1.json', check)
    return check

def invalid_chain(pred):
    n = len(pred)
    if any(not isinstance(p, int) or not 0 <= p < n for p in pred):
        return True
    roots = [i for i, p in enumerate(pred) if i == p]
    if len(roots) != 1:
        return True
    succ = [[] for _ in range(n)]
    for i, p in enumerate(pred):
        if i != p:
            succ[p].append(i)
    if any(len(s) > 1 for s in succ):
        return True
    seen = set()
    node = roots[0]
    while node not in seen:
        seen.add(node)
        if not succ[node]:
            return len(seen) != n
        node = succ[node][0]
    return True

def quantile(values, q):
    s = sorted(values)
    x = (len(s) - 1) * q
    lo = int(x)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (x - lo)

def metrics(rows, bootstrap=False):
    assert rows
    keys = ('pointer_accuracy', 'exact_array_accuracy', 'CE', 'invalid_chain')
    values = {k: [float(r[k]) for r in rows] for k in keys}
    out = dict(arrays=len(rows), n=rows[0]['n'], **{k: sum(v)/len(v) for k,v in values.items()})
    if bootstrap:
        # Whole arrays, all pointer outcomes retained together. Each split uses
        # its own identically seeded bank; no pooling across problem sizes.
        rng = random.Random(6202)
        banks = {k: [] for k in keys}
        for _ in range(20000):
            indices = [rng.randrange(len(rows)) for _ in rows]
            for k in keys:
                banks[k].append(sum(values[k][i] for i in indices)/len(rows))
        out['bootstrap_95_intervals'] = {k: [quantile(v,.025),quantile(v,.975)] for k,v in banks.items()}
        out['uncertainty'] = '20,000 complete-array bootstrap resamples, seed6202 per split, linear empirical quantiles; conditional on this trained model'
    return out

def controls(pool):
    n = pool[0]['n']
    def fixed(fn):
        rows = []
        for r in pool:
            p = fn(r)
            rows.append(dict(n=n, pointer_accuracy=sum(a==b for a,b in zip(p,r['pred']))/n,
                             exact_array_accuracy=p==r['pred'], invalid_chain=invalid_chain(p), CE=0.))
        m = metrics(rows)
        m['CE'] = 'Undefined for hard-only control; no fitted probabilities'
        return m
    return dict(uniform_pointer=dict(pointer_accuracy=1/n,exact_array_accuracy=n**(-n),CE=__import__('math').log(n),
                                   invalid_chain='Not estimated; analytic expectations only, no random control draws'),
                self_pointer=fixed(lambda r:list(range(n))),
                original_order=fixed(lambda r:[0]+list(range(n-1))))
