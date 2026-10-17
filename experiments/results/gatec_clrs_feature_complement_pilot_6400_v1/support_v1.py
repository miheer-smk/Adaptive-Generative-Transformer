"""File/provenance helpers only; no model imports."""
from pathlib import Path
import json,hashlib,os,stat,subprocess,time,shutil
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
ENG=OUT.parent/'gatec_clrs_insertion_sort_6200_v1'
def save(name,value):
    p=OUT/name;temporary=p.with_name(p.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(p)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def integrity():
    base=json.loads((OUT/'integrity_initial_v1.json').read_text());branch=json.loads((OUT/'branch_creation_v1.json').read_text());bad=[];intentional=[]
    allowed={'.git/HEAD','.git/logs/HEAD'}
    for name,r in base['files'].items():
        p=ROOT/name
        try:
            s=p.lstat();mode=stat.S_IMODE(s.st_mode)==r['mode']
            if name=='.git/HEAD':
                ok=mode and p.read_text()=='ref: refs/heads/'+branch['branch']+'\n';intentional.append(name)
            elif name=='.git/logs/HEAD':
                b=p.read_bytes();ok=mode and len(b)>=r['size'] and hashlib.sha256(b[:r['size']]).hexdigest()==r['sha256'];intentional.append(name)
            elif r['kind']=='symlink':ok=mode and p.is_symlink() and os.readlink(p)==r['target']
            else:ok=mode and stat.S_ISREG(s.st_mode) and s.st_size==r['size'] and sha(p)==r['sha256']
        except OSError:ok=False
        if not ok:bad.append(name)
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=env,text=True).strip()
    oldhead=subprocess.check_output(['git','rev-parse',base['original_branch']],cwd=ROOT,env=env,text=True).strip()
    diff=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT,env=env)).hexdigest()
    current=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,env=env,text=True).strip()
    return dict(success=not bad and head==oldhead==base['HEAD'] and diff==base['tracked_diff_sha256'] and current==branch['branch'],protected_files=len(base['files']),changed_or_missing=bad,original_branch_ref_unchanged=oldhead==base['HEAD'],HEAD_commit_unchanged=head==base['HEAD'],tracked_diff_unchanged=diff==base['tracked_diff_sha256'],pilot_branch=current,intentional_Git_metadata_updates=sorted(intentional),historical_exception_preserved=True)
