"""Complete hash-locked CPU dependency installation with no duplicate payloads."""
from pathlib import Path
import hashlib,json,subprocess,urllib.request,urllib.parse,shutil,zipfile

OUT=Path(__file__).resolve().parent
ENV=Path('/tmp/agt_clrs_cpu_6200_v1')
WHEELS=ENV/'resolved_wheels'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    if (OUT/'dependency_install_offline_v1.json').exists():raise SystemExit('STOP: CPU installation receipt exists')
    resolution=json.loads((OUT/'dependency_resolution_v1.json').read_text())
    records=[];uncompressed=0
    for item in resolution['install']:
        url=item['download_info']['url'];name=urllib.parse.unquote(url.rsplit('/',1)[1])
        expected=item['download_info']['archive_info']['hashes']['sha256']
        target=WHEELS/name;reused=target.exists()
        if not reused:
            with urllib.request.urlopen(url,timeout=120) as r,target.open('xb') as f:
                for b in iter(lambda:r.read(1048576),b''):f.write(b)
        assert sha(target)==expected,name
        with zipfile.ZipFile(target) as z:uncompressed+=sum(i.file_size for i in z.infolist())
        records.append(dict(wheel=name,size=target.stat().st_size,sha256=expected,reused_same_inode=reused))
    assert uncompressed+sum(x['size'] for x in records)<5*1024**3
    assert shutil.disk_usage(ENV).free>5*1024**3
    (OUT/'CPU_dependency_payloads_v1.json').write_text(json.dumps(dict(
        wheels=records,total_uncompressed_bytes=uncompressed,
        package_cache_disabled=True,no_AGT_environment_changes=True,
        interrupted_registry_install='Stopped before installation because pip preferred registry links over retained wheels; switched unchanged locked packages to offline installation. No model or official data-generation attempt occurred.'),indent=2)+'\n')
    args=[str(ENV/'bin/python'),'-B','-m','pip','install','--no-index','--no-cache-dir',
          '--no-compile','--require-hashes','--find-links',str(WHEELS),
          '--report',str(OUT/'dependency_install_offline_v1.json'),
          '-r',str(OUT/'cpu_requirements_locked_v1.txt')]
    with (OUT/'dependency_install_offline_v1.log').open('x') as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS: offline hash-locked CPU installation; no AGT environment changes')

if __name__=='__main__':main()
