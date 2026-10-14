"""Official sampler export and symbolic verification only; no AGT imports."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import hashlib
import importlib.metadata
import json
from pathlib import Path
import traceback
import numpy as np
from official_loader_v1 import load_official

OUT = Path(__file__).resolve().parent
os.environ['KERAS_HOME'] = str(OUT/'cpu_runtime_home/keras')
os.environ['XDG_CACHE_HOME'] = str(OUT/'cpu_runtime_home/cache')
os.environ['JAX_COMPILATION_CACHE_DIR'] = str(OUT/'cpu_runtime_home/jax_cache')
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def main():
    if (OUT / 'data_export_started_v1.json').exists():
        raise RuntimeError('No export overwrite/retry')
    write('data_export_started_v1.json', {'official_revision': '6d5728b9efadf8648413c8c473929a733caacced'})
    source = json.loads((OUT / 'official_source_manifest_v1.json').read_text())
    for r in source['files']:
        assert sha(OUT / 'official_source' / r['path']) == r['sha256']
    samplers, specs, probing, evaluation, sorting = load_official()
    assert samplers.CLRS30['train'] == dict(num_samples=1000, length=16, seed=1)
    tie_cases = [[.5, .2, .5, .1], [.3, .3], [.1, .1, .1], [0., 1.], [1., 0.]]
    verification = []
    for a in tie_cases:
        original = np.array(a, dtype=np.float64)
        _, probes = sorting.insertion_sort(original.copy())
        _, outputs, _ = probing.split_stages(probes, specs.SPECS['insertion_sort'])
        official = outputs[0].data[0]
        order = np.argsort(original, kind='stable')
        expected = np.arange(len(a))
        expected[order[1:]] = order[:-1]
        assert np.array_equal(official, expected)
        assert official[order[0]] == order[0]
        verification.append(dict(keys=a, pred=official.tolist(), stable_ties_pass=True))
    pools = {}; all_hashes = set(); manifest = []
    for split, count, n, seed in [('train',1000,16,1), ('development',32,16,2), ('size_transfer',32,64,3)]:
        sampler, spec = samplers.build_sampler('insertion_sort', seed=seed,
            num_samples=count, length=n, truncate_decimals=None)
        feedback = sampler.next(batch_size=None)
        processed_spec, iterator = samplers.process_permutations(spec, iter([feedback]), False)
        feedback = next(iterator)
        assert processed_spec['pred'] == (specs.Stage.OUTPUT, specs.Location.NODE, specs.Type.POINTER)
        inputs = {d.name:d for d in feedback.features.inputs}
        assert set(inputs) == {'pos','key'}
        assert len(feedback.outputs) == 1 and feedback.outputs[0].name == 'pred'
        key = np.asarray(inputs['key'].data)
        pos = np.asarray(inputs['pos'].data)
        pred = np.asarray(feedback.outputs[0].data)
        assert key.shape == pos.shape == pred.shape == (count,n)
        assert np.isfinite(key).all() and np.isfinite(pos).all()
        assert (pred >= 0).all() and (pred < n).all()
        key32, pos32 = key.astype(np.float32), pos.astype(np.float32)
        assert np.array_equal(pos32.astype(np.float64), pos)
        rows = []; encodings = []
        for i in range(count):
            # Validation, not target generation: original official output remains authoritative.
            order = np.argsort(key[i], kind='stable')
            assert np.array_equal(order, np.argsort(key32[i], kind='stable')), 'FP32 changes order'
            assert np.unique(key[i]).size == np.unique(key32[i]).size, 'FP32 loses distinct values'
            expected = np.arange(n); expected[order[1:]] = order[:-1]
            assert np.array_equal(pred[i], expected)
            digest = hashlib.sha256(key[i].tobytes() + pos[i].tobytes()).hexdigest()
            assert digest not in all_hashes, 'Input overlap or duplicate; no replacement'
            all_hashes.add(digest)
            row = dict(id=f'official/{split}/row{i:04d}', n=n,
                       key_original=key[i].tolist(), key_fp32=key32[i].tolist(),
                       pos=pos32[i].tolist(), pred=pred[i].astype(int).tolist(),
                       original_input_sha256=digest)
            rows.append(row)
            x = np.zeros((2*n,4), dtype=np.float32)
            x[:n,0]=1; x[:n,2]=pos32[i]; x[:n,3]=key32[i]
            x[n:,1]=1; x[n:,2]=pos32[i]
            assert not x[n:,3].any()
            encodings.append(x)
        oracle = probing.DataPoint(name='pred', location=specs.Location.NODE,
                                  type_=specs.Type.POINTER, data=pred.copy())
        score = evaluation.evaluate(feedback.outputs, {'pred':oracle})
        assert float(score['score']) == 1.0
        corrupted = pred.copy(); corrupted[0,0]=(corrupted[0,0]+1)%n
        wrong = probing.DataPoint(name='pred',location=specs.Location.NODE,
                                  type_=specs.Type.POINTER,data=corrupted)
        wrong_score = float(evaluation.evaluate(feedback.outputs, {'pred':wrong})['score'])
        assert abs(wrong_score-(1-1/(count*n))) < 1e-12
        json_name=split+'_pool_v1.json'; write(json_name, rows)
        raw = dict(key_original=key, key_fp32=key32, pos_original=pos,
                   pos_fp32=pos32, pred=pred, encoded_features=np.stack(encodings),
                   hint_lengths=feedback.features.lengths)
        for h in feedback.features.hints:raw['hint_'+h.name] = h.data
        npz_name=split+'_official_arrays_v1.npz'
        with (OUT/npz_name).open('xb') as f:np.savez_compressed(f,**raw)
        manifest.append(dict(split=split,algorithm='insertion_sort',count=count,n=n,seed=seed,
            sampler_kwargs=dict(seed=seed,num_samples=count,length=n,truncate_decimals=None),
            enforcement=False,random_pos=False,hints_used=False,oracle_score=float(score['score']),
            one_corrupted_pointer_score=wrong_score,FP32_order_and_distinctness_pass=True,
            original_dtypes=dict(key=str(key.dtype),pos=str(pos.dtype),pred=str(pred.dtype)),
            hint_schema={h.name:dict(shape=list(h.data.shape),dtype=str(h.data.dtype),type=h.type_) for h in feedback.features.hints},
            artifacts={name:dict(sha256=sha(OUT/name),size=(OUT/name).stat().st_size) for name in (json_name,npz_name)}))
        pools[split] = rows
    # Fixed engineering fixtures, not additional sampled scientific pools.
    fixtures = []
    keys_by_size = {2:[.8,.2], 4:[.5,.2,.5,.1],
        8:[.75,.25,.875,.125,.5,0.,.625,.375],
        16:[((5*i+3)%16)/16 for i in range(16)],
        64:[((13*i+7)%64)/64 for i in range(64)]}
    for n, keys in keys_by_size.items():
        _, probes = sorting.insertion_sort(np.array(keys, dtype=np.float64))
        _, outputs, _ = probing.split_stages(probes, specs.SPECS['insertion_sort'])
        fixtures.append(dict(id=f'engineering/synthetic/N{n}',n=n,key_fp32=keys,
                             pos=(np.arange(n)/n).tolist(),pred=outputs[0].data[0].astype(int).tolist(),
                             provenance='Prospectively fixed unit fixture; labels from unchanged official insertion_sort'))
    write('smoke_fixture_inputs_v1.json', fixtures)
    versions={n:importlib.metadata.version(n) for n in ['numpy','jax','jaxlib','chex','attrs','absl-py','tensorflow']}
    write('official_data_manifest_v1.json',dict(revision=source['revision'],official_sampler_unmodified=True,
        official_source_manifest_sha256=sha(OUT/'official_source_manifest_v1.json'),
        generated_samples=1064,pools=manifest,software=versions,CPU_only=True,
        TFDS_imported=False,split_overlap=0,within_split_duplicates=0,
        stable_tie_verification=verification,byte_equivalent_TFRecord_claim=False,
        task_description='Official insertion-sort input/output task; serialized AGT adapter, output-only supervision; no original GNN reproduction'))
    write('data_export_verification_v1.json',dict(success=True,target_generation=True,
        stable_ties=True,pointer_bounds=True,FP32_fidelity=True,round_trip_official_metric=True,
        duplicate_and_overlap_check=True,no_AGT_import=True,model_tokens=0))
    print('PASS: 1000 train/32 development/32 size-transfer; official sampler and evaluator')

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        write('data_export_failure_v1.json', dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        raise
