import os,json,hashlib,time,subprocess,stat
from pathlib import Path
OUT=Path(__file__).resolve().parent
CACHE=OUT.parent
ROOT=OUT.parents[3]
ENG=CACHE.parent/'gatec_clrs_insertion_sort_6200_v1'
def save(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def deadline():
    c=json.loads((OUT/'stage_clock_v1.json').read_text());return c['measured_start_monotonic']+c['guard_seconds_after_marker']
def check():
    if time.monotonic()>=deadline():raise RuntimeError('RESOURCE_STOP: remaining-stage deadline')
    if os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize<256*1024**2:raise RuntimeError('RESOURCE_STOP: storage reserve')
def integrity():
    d=json.loads((OUT/'integrity_initial_v1.json').read_text());bad=[]
    for n,r in d['files'].items():
        f=ROOT/n
        try:
            s=f.lstat();ok=(f.is_symlink() and os.readlink(f)==r['target'] and stat.S_IMODE(s.st_mode)==r['mode']) if r['kind']=='symlink' else (stat.S_ISREG(s.st_mode) and s.st_size==r['size'] and stat.S_IMODE(s.st_mode)==r['mode'] and sha(f)==r['sha256'])
            if not ok:bad.append(n)
        except Exception as e:bad.append(dict(path=n,type=type(e).__name__))
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=env,text=True).strip()
    diff=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT,env=env)).hexdigest()
    return dict(success=not bad and head==d['HEAD'] and diff==d['tracked_diff_sha256'],protected_files=len(d['files']),changed_or_missing=bad,HEAD_unchanged=head==d['HEAD'],tracked_diff_unchanged=diff==d['tracked_diff_sha256'],historical_exception_preserved=True)
