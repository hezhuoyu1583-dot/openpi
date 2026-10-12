"""CPU data/transform and masked-loss gradient checks; not a full model training run."""
import dataclasses
import json
from pathlib import Path
import numpy as np
import torch
from openpi.training import config, data_loader
from openpi import transforms
from openpi.policies.linglong_policy import PROMPT

def main():
    cfg=config.get_config('pi05_linglong_sft')
    dc=cfg.data.create(cfg.assets_dirs,cfg.model); root=Path(dc.dataset_root)
    train=json.loads((root/'train_episodes.json').read_text()); val=json.loads((root/'validation_episodes.json').read_text())
    mapping=json.loads((root/'source_mapping.json').read_text())
    assert not set(train)&set(val)
    assert not {r['source_id'] for r in mapping if r['split']=='train'} & {r['source_id'] for r in mapping if r['split']=='validation'}
    checks=[]
    for label,ids in [('train',train),('validation',val)]:
        # Exercise the real loader, including non-contiguous episode indexing.
        selected=root/f'verify_{label}_episodes.json'; selected.write_text(json.dumps([ids[0],ids[-1]]))
        ds=data_loader.create_torch_dataset(dataclasses.replace(dc,episode_indices_path=str(selected)),cfg.model.action_horizon,cfg.model)
        transform=transforms.compose([*dc.repack_transforms.inputs,*dc.data_transforms.inputs,
            transforms.Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs])
        for idx in (0,len(ds)-1):
            sample=ds[idx]; assert sample['prompt']==PROMPT
            item=transform(sample)
            assert item['state'].shape==(32,)
            assert item['actions'].shape==(cfg.model.action_horizon,32)
            assert np.isfinite(item['actions']).all()
            assert item['action_dim_mask'].sum()==22
            assert all(im.shape==(224,224,3) for im in item['image'].values())
            # Same dimension reduction as train_pytorch. Verify ignored outputs get no gradient.
            pred=torch.ones((1,cfg.model.action_horizon,32),requires_grad=True)
            target=torch.tensor(item['actions'])[None]
            mask=torch.tensor(item['action_dim_mask'],dtype=torch.float32)[None]
            loss=(((pred-target)**2)*mask[:,None]).sum(-1)/mask.sum(-1,keepdim=True)
            loss.mean().backward()
            assert torch.count_nonzero(pred.grad[:,:,22:])==0
            checks.append({'split':label,'index':idx,'loss':float(loss.mean().detach()),'shape':list(item['actions'].shape)})
    result={'status':'passed','checks':checks,'prompt':PROMPT,
            'gpu_used':False,'full_model_forward_backward_tested':False}
    loader=data_loader.create_data_loader(dataclasses.replace(cfg,batch_size=2,num_workers=0),
        framework='pytorch',num_batches=1,shuffle=False)
    observation,actions=next(iter(loader))
    assert tuple(actions.shape)==(2,cfg.model.action_horizon,32)
    assert tuple(observation.state.shape)==(2,32)
    assert observation.action_dim_mask.sum().item()==44
    result['full_training_loader_batch']={'state':list(observation.state.shape),'actions':list(actions.shape)}
    # Validate checkpoint shapes and shared-parameter aliases without allocating model weights/GPU memory.
    from safetensors import safe_open
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    with torch.device('meta'): model=PI0Pytorch(cfg.model)
    expected=model.state_dict(); named=dict(model.named_parameters(remove_duplicate=False))
    with safe_open(str(Path(cfg.pytorch_weight_path)/'model.safetensors'),framework='pt',device='cpu') as f:
        keys=set(f.keys()); missing=set(expected)-keys
        assert not keys-set(expected)
        assert all(list(expected[k].shape)==f.get_slice(k).get_shape() for k in keys)
        assert all(any(other in keys and param is named[k] for other,param in named.items()) for k in missing)
        result['checkpoint_shape_check']={'stored_tensors':len(keys),'shared_aliases_omitted':sorted(missing),'status':'passed'}
    (root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
