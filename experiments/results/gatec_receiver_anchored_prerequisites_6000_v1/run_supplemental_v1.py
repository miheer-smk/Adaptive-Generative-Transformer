"""Bounded supplemental fixtures; preserves every previous execution result."""
import os,sys,json,hashlib,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(OUT),str(ROOT)]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
os.environ['CUDA_CACHE_PATH']=str(OUT/'runtime_cache')
from run_prerequisites_v3 import verify_integrity
before=json.loads((OUT/'integrity_before.json').read_text())
assert verify_integrity(before)['success']
gate=OUT/'supplemental_started.json'
if gate.exists(): raise SystemExit('STOP: supplemental fixtures already started')
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')}
gate.write_text(json.dumps(dict(source_hashes=sources,additional_model_tokens=8,bootstrap_only_other_tests=True),indent=2)+'\n')
import torch,pytest
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
status=1
try:
 status=int(pytest.main([str(OUT/'test_supplemental_boundaries_v1.py'),'-q','-x','-p','no:cacheprovider',
                       '--confcutdir='+str(OUT),'--junitxml='+str(OUT/'supplemental_results.xml')]))
finally:
 final=verify_integrity(before)
 (OUT/'integrity_after_supplemental.json').write_text(json.dumps(final,indent=2)+'\n')
 (OUT/'supplemental_execution.json').write_text(json.dumps(dict(pytest_exit_code=status,integrity=final['success'],
                                                             optimizer_steps=0),indent=2)+'\n')
raise SystemExit(status)
