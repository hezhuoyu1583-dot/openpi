"""CPU verification of 0923 Parquet, split isolation, repair and real OpenPI loading."""
import csv
import dataclasses
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from openpi.training import config, data_loader
from openpi import transforms
from prepare_0923 import ROOT,TRAIN_OUT,TEST_OUT,PROMPT
from prepare_sft import write

def main():
    inventories={s:json.loads((ROOT/s/'inventory.json').read_text()) for s in ('train','test')}
    for key in ('tar_sha256','db_sha256','images_sha256'):
        assert not {r[key] for r in inventories['train']} & {r[key] for r in inventories['test']}
    assert len(inventories['test'])==10
    cfg=config.get_config('pi05_linglong_0923_sft')
    assert cfg.policy_metadata['prompt']==PROMPT
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    results=[]
    for split,root in [('train',TRAIN_OUT),('test',TEST_OUT)]:
        mapping=json.loads((root/'source_mapping.json').read_text())
        assert len(mapping)==len(inventories[split]), 'some episodes rejected; inspect before declaring complete'
        assert json.loads((root/'meta/tasks.jsonl').read_text())['task']==PROMPT
        frames=0; changed=0
        for r in mapping:
            i=r['episode_index']; n=r['length'];source=ROOT/split/'decoded'/r['source_id']
            table=pq.read_table(root/f'data/chunk-000/episode_{i:06d}.parquet')
            assert len(table)==n
            rows=list(csv.DictReader((source/'aligned.csv').open()))
            indices=r['source_row_indices']
            for column,prefix in [('observation.state','q'),('action','q_exp')]:
                actual=np.array(table[column].to_pylist(),dtype=np.float32)
                expected=np.array([[float(rows[j][f'{prefix}_{k}']) for k in range(24)] for j in indices],dtype=np.float32)
                assert np.array_equal(actual,expected)
            assert np.allclose(np.array(table['timestamp']) ,np.arange(n)/15,atol=1e-6)
            for cam in ('head','left','right'):
                assert all(Path(x['path']).is_file() for x in table['observation.images.'+cam].to_pylist())
            frames+=n
        manifest=root/('train_episodes.json' if split=='train' else 'test_episodes.json')
        split_dc=dataclasses.replace(dc,repo_id='linglong_0923_'+split,dataset_root=str(root),episode_indices_path=str(manifest))
        ds=data_loader.create_torch_dataset(split_dc,25,cfg.model)
        assert len(ds)==frames
        pipeline=transforms.compose([*dc.repack_transforms.inputs,*dc.data_transforms.inputs,
            transforms.Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs])
        offset=0
        for rec in mapping:
            # Each episode's beginning and tail exercise real images, prompt, and action padding.
            for idx in (offset,offset+rec['length']-1):
                sample=ds[idx];assert sample['prompt']==PROMPT
                item=pipeline(sample)
                assert item['state'].shape==(32,) and item['actions'].shape==(25,32)
                assert np.isfinite(item['actions']).all() and np.isfinite(item['state']).all()
                assert item['action_dim_mask'].sum()==22
                assert all(im.shape==(224,224,3) for im in item['image'].values())
            offset+=rec['length']
        results.append(dict(split=split,episodes=len(mapping),frames=frames,
                            full_25_step_windows=sum(r['length']-24 for r in mapping)))
    assert (TRAIN_OUT/'assets/linglong_h/norm_stats.json').read_bytes()==(TEST_OUT/'assets/linglong_h/norm_stats.json').read_bytes()
    # Recompute normalization exclusively from final train parquet.
    stats=json.loads((TRAIN_OUT/'assets/linglong_h/norm_stats.json').read_text())['norm_stats']
    for k,column in [('state','observation.state'),('actions','action')]:
        a=np.concatenate([np.array(pq.read_table(f,columns=[column])[column].to_pylist(),dtype=np.float64)
                          for f in sorted((TRAIN_OUT/'data').rglob('*.parquet'))])
        for name,value in [('mean',a.mean(0)),('std',a.std(0)),('q01',np.quantile(a,.01,axis=0)),('q99',np.quantile(a,.99,axis=0))]:
            assert np.allclose(value,stats[k][name],atol=1e-10)
    result=dict(status='passed',prompt=PROMPT,splits=results,split_hash_overlap=0,
        train_only_normalization_verified=True,all_episode_boundary_samples_transformed=True,
        state_shape=[32],action_shape=[25,32],valid_action_dims=22,gpu_used=False)
    write(ROOT/'verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
