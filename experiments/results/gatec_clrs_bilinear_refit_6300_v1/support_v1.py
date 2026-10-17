"""Research-only file/integrity helpers; no model imports."""
import hashlib,json,os,stat,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
CACHE=OUT.parent/'gatec_clrs_frozen_readout_cache_6300_v1'
PRIOR=CACHE/'decodability_results_v1'
ENG=OUT.parent/'gatec_clrs_insertion_sort_6200_v1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def check():
    assert os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize>256*1024**2,'Storage reserve'
def integrity():
    d=json.loads((OUT/'integrity_initial_v1.json').read_text());bad=[]
    for n,r in d['files'].items():
        p=ROOT/n
        try:
            s=p.lstat();ok=stat.S_IMODE(s.st_mode)==r['mode']
            if r['kind']=='file':ok=ok and stat.S_ISREG(s.st_mode) and s.st_size==r['size'] and sha(p)==r['sha256']
            else:ok=ok and p.is_symlink() and os.readlink(p)==r['target']
        except OSError:ok=False
        if not ok:bad.append(n)
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=env,text=True).strip()
    diff=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT,env=env)).hexdigest()
    return dict(success=not bad and head==d['HEAD'] and diff==d['tracked_diff_sha256'],protected_files=len(d['files']),changed_or_missing=bad,HEAD_unchanged=head==d['HEAD'],tracked_diff_unchanged=diff==d['tracked_diff_sha256'],historical_exception_preserved=True)
