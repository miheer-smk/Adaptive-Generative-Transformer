"""Single bounded publication recovery; immutable prior failure retained.

Reuses uploaded Git blobs. Raw blob streaming checks avoid the interrupted
large base64 JSON response. No model imports, local Git writes or payload copies.
"""
import base64,hashlib,http.client,json,os,re,subprocess,time,urllib.request,urllib.error
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
REPO='miheer-smk/Adaptive-Generative-Transformer'
API='https://api.github.com/repos/'+REPO
TOKEN=subprocess.check_output(['gh','auth','token'],stderr=subprocess.DEVNULL).decode().strip()
HEADERS={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'AGT-publication-verifier'}
LAST=0.;RETRIES=[]

def api(path,method='GET',data=None):
    global LAST
    if method!='GET':
        wait=max(0,LAST+8-time.monotonic())
        if wait:time.sleep(wait)
        LAST=time.monotonic()
    req=urllib.request.Request(API+path,None if data is None else json.dumps(data,separators=(',',':')).encode(),{**HEADERS,'Content-Type':'application/json'},method=method)
    # Mutations are not retried after ambiguous transport failure.
    for attempt in range(3 if method=='GET' else 1):
        try:
            with urllib.request.urlopen(req,timeout=180) as f:return json.loads(f.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError('HTTP status '+str(e.code)+' at '+path) from None
        except (http.client.IncompleteRead,TimeoutError,urllib.error.URLError,json.JSONDecodeError) as e:
            if method!='GET' or attempt==2:raise RuntimeError(type(e).__name__+' at '+path) from None
            RETRIES.append(dict(path=path,attempt=attempt+1,type=type(e).__name__))
            time.sleep(10)

def records(tree):
    x=api('/git/trees/'+tree+'?recursive=1');assert not x.get('truncated')
    return {r['path']:r for r in x['tree'] if r['type']=='blob'}

def git_oid(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

def upload(data):
    v=api('/git/blobs','POST',dict(encoding='base64',content=base64.b64encode(data).decode()))
    assert v['sha']==git_oid(data);return v['sha']

def verify_raw(r):
    for attempt in range(3):
        try:
            sh=hashlib.sha256();gh=hashlib.sha1(b'blob '+str(r['size']).encode()+b'\0');size=0
            req=urllib.request.Request(API+'/git/blobs/'+r['git_oid'],headers={**HEADERS,'Accept':'application/vnd.github.raw+json'})
            with urllib.request.urlopen(req,timeout=180) as f:
                while True:
                    chunk=f.read(1048576)
                    if not chunk:break
                    size+=len(chunk);assert size<=r['size'];sh.update(chunk);gh.update(chunk)
            if size!=r['size']:raise http.client.IncompleteRead(b'',r['size']-size)
            assert sh.hexdigest()==r['sha256'] and gh.hexdigest()==r['git_oid'],r['path']
            return
        except (http.client.IncompleteRead,TimeoutError,urllib.error.URLError) as e:
            if attempt==2:raise RuntimeError('Raw verification failed at '+r['path']) from None
            RETRIES.append(dict(path=r['path'],attempt=attempt+1,type=type(e).__name__))
            time.sleep(10)

def main():
    marker=OUT/'publication_started_v1.json'
    if marker.exists():raise RuntimeError('Diagnostic publication already attempted')
    assert json.loads((OUT/'numeric_verification_v1.json').read_text())['success']
    status='COMPLETED_OFFLINE_AUDIT'
    assert json.loads((OUT/'integrity_final_v1.json').read_text())['success']
    parent_sha='f40f65d6c4008f407669085cd5655ccc52f3424c'
    branch='research/clrs-residual-score-audit-6200-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    try:api('/git/ref/heads/'+branch)
    except RuntimeError as e:
        if not str(e).startswith('HTTP status 404 '):raise
    else:raise RuntimeError('Remote branch exists; no overwrite permitted')
    marker.write_text(json.dumps(dict(branch=branch,parent=parent_sha,status=status,method='Git Database API; streamed raw verification; no local staging'),indent=2)+'\n')
    parent=api('/git/commits/'+parent_sha);existing=records(parent['tree']['sha'])
    current=[]
    patterns=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{36,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),re.compile(rb'AKIA[0-9A-Z]{16}')]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file():continue
        assert not p.is_symlink()
        b=p.read_bytes();rel=p.relative_to(ROOT).as_posix()
        assert rel not in existing and len(b)<=100*1024**2
        assert not any(v.search(b) for v in patterns),'Credential signature detected; value not logged'
        current.append(dict(path=rel,size=len(b),sha256=hashlib.sha256(b).hexdigest(),git_oid=git_oid(b),mode='100755' if os.access(p,os.X_OK) else '100644'))
    catalog=OUT/'publication_inventory_v1.json'
    catalog.write_text(json.dumps(dict(files=current,self_excluded=True,LFS_required=False,secret_scan='Common credential/private-key signatures; heuristic limitation disclosed'),indent=2)+'\n')
    b=catalog.read_bytes();current.append(dict(path=catalog.relative_to(ROOT).as_posix(),size=len(b),sha256=hashlib.sha256(b).hexdigest(),git_oid=git_oid(b),mode='100644'))
    entries=[]
    for i,r in enumerate(current,1):
        assert upload((ROOT/r['path']).read_bytes())==r['git_oid']
        verify_raw(r)
        entries.append(dict(path=r['path'],mode=r['mode'],type='blob',sha=r['git_oid']))
        print('Uploaded and verified',i,'/',len(current),r['path'],flush=True)
    tree=api('/git/trees','POST',dict(base_tree=parent['tree']['sha'],tree=entries))
    actual=records(tree['sha'])
    for name,r in existing.items():assert actual[name]['sha']==r['sha'] and actual[name]['mode']==r['mode']
    for r in current:assert actual[r['path']]['sha']==r['git_oid'] and actual[r['path']]['mode']==r['mode']
    receipt=OUT/'remote_verification_v1.json'
    receipt.write_text(json.dumps(dict(repository='https://github.com/'+REPO,branch=branch,parent=parent_sha,status=status,verified_files=len(current),verified_tree=tree['sha'],historical_remote_files_preserved=len(existing),verification='Every raw payload streamed; size, Git OID and SHA256 match',retries=RETRIES,LFS_objects=0,omitted_files=[],final_commit_identity='Printed after final ref verification; receipt included without circular self-hashing'),indent=2)+'\n')
    b=receipt.read_bytes();rr=dict(path=receipt.relative_to(ROOT).as_posix(),size=len(b),sha256=hashlib.sha256(b).hexdigest(),git_oid=upload(b),mode='100644')
    final_tree=api('/git/trees','POST',dict(base_tree=tree['sha'],tree=[dict(path=rr['path'],mode='100644',type='blob',sha=rr['git_oid'])]))
    final=records(final_tree['sha']);assert len(final)==len(existing)+len(current)+1
    for name,r in actual.items():assert final[name]['sha']==r['sha'] and final[name]['mode']==r['mode']
    verify_raw(rr)
    commit=api('/git/commits','POST',dict(message='Archive offline CLRS residual geometry and target-conditioned score audit: zero model calls',tree=final_tree['sha'],parents=[parent_sha]))
    api('/git/refs','POST',dict(ref='refs/heads/'+branch,sha=commit['sha']))
    ref=api('/git/ref/heads/'+branch);assert ref['object']['sha']==commit['sha']
    c=api('/git/commits/'+commit['sha']);assert c['tree']['sha']==final_tree['sha'] and [x['sha'] for x in c['parents']]==[parent_sha]
    for r in current+[rr]:
        b=(ROOT/r['path']).read_bytes();assert len(b)==r['size'] and hashlib.sha256(b).hexdigest()==r['sha256']
    print(json.dumps(dict(repository='https://github.com/'+REPO,branch=branch,commit=commit['sha'],parent=parent_sha,expected_files=len(current)+1,published_files=len(current)+1,verified_files=len(current)+1,all_payloads_verified=True,LFS_objects=0,omitted_files=[]),indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        (OUT/'publication_failure_v1.json').write_text(json.dumps(dict(type=type(e).__name__,message=str(e) if isinstance(e,RuntimeError) else 'Publication failed; credentials not logged'),indent=2))
        raise SystemExit('Publication stopped: '+type(e).__name__)
