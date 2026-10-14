"""One fresh remote snapshot, no local Git staging or object writes.

Verify an unreferenced candidate commit, include that verification receipt in
the final published snapshot, then verify the final ref. This avoids a receipt
that would need to contain its own commit hash. All evidence stays in this run.
"""
import base64,hashlib,json,os,re,subprocess,time,urllib.request,urllib.error
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
REPO='miheer-smk/Adaptive-Generative-Transformer'
API='https://api.github.com/repos/'+REPO
PARENT='db0e62ca0fad464fd840ac2754eb759bf53bb44b'
TOKEN=subprocess.check_output(['gh','auth','token'],stderr=subprocess.DEVNULL).decode().strip()
HEADERS={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json',
         'X-GitHub-Api-Version':'2022-11-28','User-Agent':'AGT-engineering-milestone-publisher'}
LAST=0.

def api(path,method='GET',data=None):
    global LAST
    if method!='GET':
        wait=max(0,LAST+8-time.monotonic())
        if wait:time.sleep(wait)
        LAST=time.monotonic()
    body=None if data is None else json.dumps(data,separators=(',',':')).encode()
    request=urllib.request.Request(API+path,body,{**HEADERS,'Content-Type':'application/json'},method=method)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
        except urllib.error.HTTPError as e:
            if e.code in (429,500,502,503,504) and attempt<2:
                wait=int(e.headers.get('Retry-After','30'))
                if wait>300:raise RuntimeError('Rate-limit delay exceeds bounded publication retry') from None
                time.sleep(wait);continue
            raise RuntimeError('HTTP status '+str(e.code)) from None

def blob_oid(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
def blob(data):
    result=api('/git/blobs','POST',{'encoding':'base64','content':base64.b64encode(data).decode()})
    assert result['sha']==blob_oid(data)
    return result['sha']
def remote_blob(sha):
    result=api('/git/blobs/'+sha);assert result['encoding']=='base64'
    data=base64.b64decode(result['content']);assert len(data)==result['size'] and blob_oid(data)==sha
    return data
def tree_records(tree):
    result=api('/git/trees/'+tree+'?recursive=1');assert not result.get('truncated')
    return {r['path']:r for r in result['tree'] if r['type']=='blob'}

def main():
    marker=OUT/'publication_started_v1.json'
    if marker.exists():raise SystemExit('STOP: no automatic publication restart')
    assert json.loads((OUT/'final_results_v1.json').read_text())['engineering_pass']
    assert json.loads((OUT/'integrity_final_v1.json').read_text())['success']
    branch='research/clrs-insertion-sort-engineering-6200-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    try:api('/git/ref/heads/'+branch)
    except RuntimeError as e:
        if str(e)!='HTTP status 404':raise
    else:raise RuntimeError('New remote branch already exists')
    marker.write_text(json.dumps(dict(repository='https://github.com/'+REPO,branch=branch,parent=PARENT,
        method='Git Database API, no local staging',no_overwrite=True),indent=2)+'\n')
    parent=api('/git/commits/'+PARENT);existing=tree_records(parent['tree']['sha'])
    attrs=remote_blob(existing['.gitattributes']['sha']).decode() if '.gitattributes' in existing else ''
    # Previous archive uses exact-path LFS rules. Broad rules require an explicit
    # reviewed LFS upload path; this small milestone has no oversized files.
    for line in attrs.splitlines():
        if 'filter=lfs' in line and any(c in line.split()[0] for c in '*?['):
            raise RuntimeError('Broad inherited LFS attributes require reviewed handling')
    records=[];findings=[]
    patterns=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
              re.compile(rb'gh[pousr]_[A-Za-z0-9]{36,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
              re.compile(rb'AKIA[0-9A-Z]{16}')]
    for f in sorted(OUT.rglob('*')):
        if not f.is_file():continue
        if f.is_symlink():raise RuntimeError('Unexpected study symlink')
        data=f.read_bytes();rel=f.relative_to(ROOT).as_posix()
        if len(data)>100*1024**2:raise RuntimeError('Oversized payload needs verified LFS; stop before commit')
        if any(regex.search(data) for regex in patterns):findings.append(rel)
        if rel in existing:raise RuntimeError('Milestone path already exists in selected parent')
        records.append(dict(path=rel,size=len(data),sha256=hashlib.sha256(data).hexdigest(),
            git_oid=blob_oid(data),mode='100755' if os.access(f,os.X_OK) else '100644'))
    if findings:
        (OUT/'publication_security_blocker_v1.json').write_text(json.dumps({'paths':findings,'values_not_logged':True}))
        raise RuntimeError('Credential signature detected; affected content not published')
    catalog=OUT/'publication_input_inventory_v1.json'
    catalog.write_text(json.dumps(dict(files=records,self_excluded=True,LFS_required=False,
        secret_scan='New artifact contents checked for common private-key/GitHub/AWS signatures; heuristic, not proof of absence'),indent=2)+'\n')
    data=catalog.read_bytes();records.append(dict(path=catalog.relative_to(ROOT).as_posix(),size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),git_oid=blob_oid(data),mode='100644'))
    entries=[]
    for i,r in enumerate(records,1):
        data=(ROOT/r['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==r['sha256']
        sh=blob(data);entries.append({'path':r['path'],'mode':r['mode'],'type':'blob','sha':sh})
        print('Uploaded',i,'/',len(records),r['path'],flush=True)
    tree=api('/git/trees','POST',{'base_tree':parent['tree']['sha'],'tree':entries})
    candidate=api('/git/commits','POST',{'message':'Verify CLRS insertion-sort engineering evidence before publication',
        'tree':tree['sha'],'parents':[PARENT]})
    actual=tree_records(tree['sha'])
    for name,old in existing.items():assert actual[name]['sha']==old['sha'] and actual[name]['mode']==old['mode']
    for r in records:
        assert actual[r['path']]['sha']==r['git_oid'] and actual[r['path']]['mode']==r['mode']
        payload=remote_blob(r['git_oid'])
        assert len(payload)==r['size'] and hashlib.sha256(payload).hexdigest()==r['sha256']
    receipt=OUT/'remote_candidate_verification_v1.json'
    receipt.write_text(json.dumps(dict(repository='https://github.com/'+REPO,branch=branch,
        candidate_commit=candidate['sha'],candidate_tree=tree['sha'],parent=PARENT,
        candidate_is_unreferenced=True,verified_artifact_files=len(records),
        verification='Downloaded each remote blob into memory; checked size, Git OID and SHA-256; no disk payload copies',
        historical_remote_files_preserved=len(existing),LFS_required=False,omitted_files=[],
        final_commit_identity='Reported in terminal after final branch verification; receipt is included in final commit without circular self-hashing'),indent=2)+'\n')
    receipt_data=receipt.read_bytes();receipt_entry={'path':receipt.relative_to(ROOT).as_posix(),
        'mode':'100644','type':'blob','sha':blob(receipt_data)}
    final_tree=api('/git/trees','POST',{'base_tree':tree['sha'],'tree':[receipt_entry]})
    final_commit=api('/git/commits','POST',{'message':'Archive CLRS insertion-sort engineering milestone 6200 v1: ENGINEERING_PASSED',
        'tree':final_tree['sha'],'parents':[PARENT]})
    all_records=tree_records(final_tree['sha'])
    assert len(all_records)==len(existing)+len(records)+1
    for name,value in actual.items():assert all_records[name]['sha']==value['sha'] and all_records[name]['mode']==value['mode']
    assert remote_blob(receipt_entry['sha'])==receipt_data
    api('/git/refs','POST',{'ref':'refs/heads/'+branch,'sha':final_commit['sha']})
    ref=api('/git/ref/heads/'+branch);assert ref['object']['sha']==final_commit['sha']
    commit=api('/git/commits/'+final_commit['sha'])
    assert commit['tree']['sha']==final_tree['sha'] and [x['sha'] for x in commit['parents']]==[PARENT]
    for r in records:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    assert receipt.read_bytes()==receipt_data
    print(json.dumps(dict(repository='https://github.com/'+REPO,branch=branch,commit=final_commit['sha'],
        parent=PARENT,expected_files=len(records)+1,published_files=len(records)+1,verified_files=len(records)+1,
        all_remote_payload_SHA256_verified=True,LFS_objects=0,omitted_files=[],
        local_Git_state_untouched=True),indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        (OUT/'publication_failure_v1.json').write_text(json.dumps(dict(type=type(e).__name__,
            message=str(e) if isinstance(e,RuntimeError) else 'Publication failed; no credentials printed'),indent=2))
        raise SystemExit('Publication stopped: '+type(e).__name__)
