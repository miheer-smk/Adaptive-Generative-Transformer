"""Publish this completed exploratory refit only; no local Git writes/model imports."""
import base64,hashlib,json,os,re,subprocess,time,urllib.request,urllib.error,http.client
from pathlib import Path
from support_v1 import OUT,ROOT,sha,save,integrity,check
REPO='miheer-smk/Adaptive-Generative-Transformer'
API='https://api.github.com/repos/'+REPO
TOKEN=subprocess.check_output(['gh','auth','token'],stderr=subprocess.DEVNULL).decode().strip()
HEADERS={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'AGT-exploratory-refit-publisher'}
LAST=0.;RETRIES=[]
def api(path,method='GET',data=None):
    global LAST
    check()
    if method!='GET':
        delay=max(0,LAST+8-time.monotonic())
        if delay:time.sleep(delay)
        LAST=time.monotonic()
    request=urllib.request.Request(API+path,None if data is None else json.dumps(data,separators=(',',':')).encode(),{**HEADERS,'Content-Type':'application/json'},method=method)
    for i in range(3 if method=='GET' else 1):
        try:
            with urllib.request.urlopen(request,timeout=120) as f:return json.load(f)
        except urllib.error.HTTPError as e:raise RuntimeError('HTTP '+str(e.code)+' at '+path) from None
        except (http.client.IncompleteRead,TimeoutError,urllib.error.URLError,json.JSONDecodeError) as e:
            if method!='GET' or i==2:raise RuntimeError(type(e).__name__+' at '+path) from None
            RETRIES.append(dict(path=path,attempt=i+1,type=type(e).__name__));time.sleep(10)
def oid(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def record(p):
    b=p.read_bytes()
    return dict(path=p.relative_to(ROOT).as_posix(),size=len(b),sha256=hashlib.sha256(b).hexdigest(),git_oid=oid(b),mode='100755' if os.access(p,os.X_OK) else '100644')
def upload(r):
    b=(ROOT/r['path']).read_bytes()
    assert len(b)==r['size'] and hashlib.sha256(b).hexdigest()==r['sha256']
    assert api('/git/blobs','POST',dict(encoding='base64',content=base64.b64encode(b).decode()))['sha']==r['git_oid']
def verify(r):
    for attempt in range(3):
        try:
            h=hashlib.sha256();g=hashlib.sha1(b'blob '+str(r['size']).encode()+b'\0');length=0
            req=urllib.request.Request(API+'/git/blobs/'+r['git_oid'],headers={**HEADERS,'Accept':'application/vnd.github.raw+json'})
            with urllib.request.urlopen(req,timeout=120) as f:
                while True:
                    b=f.read(1048576)
                    if not b:break
                    length+=len(b);assert length<=r['size'];h.update(b);g.update(b)
            if length!=r['size']:raise http.client.IncompleteRead(b'',r['size']-length)
            assert h.hexdigest()==r['sha256'] and g.hexdigest()==r['git_oid'];return
        except (http.client.IncompleteRead,TimeoutError,urllib.error.URLError) as e:
            if attempt==2:raise RuntimeError('Raw verification failed: '+r['path']) from None
            RETRIES.append(dict(path=r['path'],attempt=attempt+1,type=type(e).__name__));time.sleep(10)
def tree_records(tree):
    data=api('/git/trees/'+tree+'?recursive=1');assert not data.get('truncated')
    return {r['path']:r for r in data['tree'] if r['type']=='blob'}
def main():
    assert not (OUT/'publication_started_v1.json').exists(),'No publication overwrite/retry'
    assert json.loads((OUT/'final_results_v1.json').read_text())['classification']=='CERTIFIED_FIXED_OBJECTIVE_NON_SUCCESS'
    assert integrity()['success']
    parent='94a7b19079dad3fd21afc801f62d65ed9101b613'
    branch='research/clrs-bilinear-refit-6300-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    try:api('/git/ref/heads/'+branch)
    except RuntimeError as e:
        if not str(e).startswith('HTTP 404 '):raise
    else:raise RuntimeError('Remote branch already exists')
    save('publication_started_v1.json',dict(branch=branch,parent=parent,method='Git Database API; no local staging, objects or ref changes'))
    original=api('/git/commits/'+parent);old=tree_records(original['tree']['sha'])
    patterns=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{36,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),re.compile(rb'AKIA[0-9A-Z]{16}')]
    rows=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file():continue
        assert not p.is_symlink();r=record(p)
        assert r['path'] not in old and r['size']<100*1024**2
        b=p.read_bytes();assert not any(x.search(b) for x in patterns),'Secret signature detected; value not printed'
        rows.append(r)
    save('publication_inventory_v1.json',dict(files=rows,self_excluded=True,secret_scan='Common key/token signatures; heuristic, not universal proof',LFS_required=False))
    rows.append(record(OUT/'publication_inventory_v1.json'))
    entries=[]
    for i,r in enumerate(rows,1):
        upload(r);verify(r)
        entries.append(dict(path=r['path'],type='blob',mode=r['mode'],sha=r['git_oid']))
        print('Uploaded and content-verified',i,'/',len(rows),r['path'],flush=True)
    tree=api('/git/trees','POST',dict(base_tree=original['tree']['sha'],tree=entries))
    actual=tree_records(tree['sha'])
    for n,r in old.items():assert actual[n]['sha']==r['sha'] and actual[n]['mode']==r['mode']
    for r in rows:assert actual[r['path']]['sha']==r['git_oid'] and actual[r['path']]['size']==r['size']
    save('remote_verification_v1.json',dict(repository='https://github.com/'+REPO,branch=branch,parent=parent,scientific_status='CERTIFIED_FIXED_OBJECTIVE_NON_SUCCESS',raw_files_verified_before_receipt=len(rows),final_file_count_including_receipt=len(rows)+1,preserved_parent_paths=len(old),verification='All raw content SHA256, size, Git OID and tree modes checked; receipt separately uploaded and raw-verified',final_commit='Reported after final branch-ref verification in terminal; no circular commit self-hash',LFS_objects=0,omissions=[],retries=RETRIES,local_integrity=integrity()))
    receipt=record(OUT/'remote_verification_v1.json');upload(receipt);verify(receipt);rows.append(receipt)
    finaltree=api('/git/trees','POST',dict(base_tree=tree['sha'],tree=[dict(path=receipt['path'],type='blob',mode=receipt['mode'],sha=receipt['git_oid'])]))
    actual=tree_records(finaltree['sha'])
    expected={p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    assert {n for n in actual if n.startswith(OUT.relative_to(ROOT).as_posix()+'/')}==expected
    for r in rows:assert actual[r['path']]['sha']==r['git_oid'] and actual[r['path']]['mode']==r['mode'] and actual[r['path']]['size']==r['size']
    for n,r in old.items():assert actual[n]['sha']==r['sha'] and actual[n]['mode']==r['mode']
    commit=api('/git/commits','POST',dict(message='Record single certified exploratory frozen-core bilinear refit; no AGT execution',tree=finaltree['sha'],parents=[parent]))
    api('/git/refs','POST',dict(ref='refs/heads/'+branch,sha=commit['sha']))
    assert api('/git/ref/heads/'+branch)['object']['sha']==commit['sha']
    remote=api('/git/commits/'+commit['sha']);assert remote['tree']['sha']==finaltree['sha'] and [v['sha'] for v in remote['parents']]==[parent]
    finalintegrity=integrity();assert finalintegrity['success']
    print(json.dumps(dict(repository='https://github.com/'+REPO,branch=branch,commit=commit['sha'],parent=parent,files_verified=len(rows),payload_bytes=sum(r['size'] for r in rows),raw_SHA256_and_tree_verification='PASS',LFS_objects=0,omissions=[],local_integrity=finalintegrity,AGT_model_calls=0,new_fits_after_certification=0),indent=2),flush=True)
if __name__=='__main__':main()
