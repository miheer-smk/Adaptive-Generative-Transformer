"""Official evaluator on retained synthetic fixture predictions; no AGT model."""
import os
from pathlib import Path
OUT=Path(__file__).resolve().parent
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['KERAS_HOME']=str(OUT/'cpu_runtime_home/keras')
os.environ['XDG_CACHE_HOME']=str(OUT/'cpu_runtime_home/cache')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json,numpy as np
from official_loader_v1 import load_official

def main():
    _,specs,probing,evaluation,_=load_official()
    result=json.loads((OUT/'smoke_results_v1.json').read_text());checked=[]
    for row in result['records']:
        if 'predictions' not in row:continue
        truth=probing.DataPoint(name='pred',location=specs.Location.NODE,
            type_=specs.Type.POINTER,data=np.array([row['truth']]))
        prediction=probing.DataPoint(name='pred',location=specs.Location.NODE,
            type_=specs.Type.POINTER,data=np.array([row['predictions']]))
        score=float(evaluation.evaluate((truth,),{'pred':prediction})['score'])
        expected=sum(a==b for a,b in zip(row['truth'],row['predictions']))/row['n']
        assert score==expected
        checked.append(dict(device=row['device'],id=row['id'],score_matches_official=True))
    with (OUT/'smoke_official_metric_verification_v1.json').open('x') as f:
        json.dump(dict(success=True,synthetic_fixtures=checked,model_tokens=0,
            official_pool_model_evaluation=False,description='Only metric-conversion verification, not task competence'),f,indent=2)
    print('PASS: retained synthetic predictions agree with official pointer evaluator')

if __name__=='__main__':main()
