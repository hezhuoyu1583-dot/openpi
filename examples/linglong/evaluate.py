"""Post-training held-out checkpoint evaluation; no robot or policy server."""
import argparse
import dataclasses
import json
from pathlib import Path
import time
import numpy as np
from openpi.training import config, data_loader
from openpi.policies import policy_config

def metrics(pred,target):
    error=np.abs(pred-target)
    groups={'left_arm':list(range(7)),'right_arm':list(range(8,15)),
            'left_gripper':[7],'right_gripper':[15],'waist':list(range(16,20)),'head':[20,21]}
    return {k:float(error[...,idx].mean()) for k,idx in groups.items()}

def main(args):
    cfg=config.get_config('pi05_linglong_sft'); dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    root=Path(dc.dataset_root)
    dc=dataclasses.replace(dc,episode_indices_path=str(root/'validation_episodes.json'))
    dataset=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
    # Fixed origins, full windows only: exclude endpoint padding from evaluation.
    valid=[]; offset=0
    mapping=json.loads((root/'source_mapping.json').read_text())
    selected=set(json.loads((root/'validation_episodes.json').read_text()))
    for r in mapping:
        if r['episode_index'] not in selected: continue
        valid.extend(range(offset,offset+r['length']-cfg.model.action_horizon+1))
        offset+=r['length']
    rng=np.random.default_rng(args.seed)
    indices=sorted(rng.choice(valid,min(args.samples,len(valid)),replace=False).tolist())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'sample_indices.json').write_text(json.dumps(indices))
    checkpoints=sorted((p for p in args.checkpoints.iterdir() if p.is_dir() and p.name.isdigit()
                        and (p/'model.safetensors').is_file()),key=lambda p:int(p.name))
    if not checkpoints: raise ValueError('no saved checkpoints')
    for checkpoint in checkpoints:
        contract=json.loads((checkpoint/'inference_config.json').read_text())
        if contract['horizon']!=cfg.model.action_horizon or contract['config_name']!=cfg.name:
            raise ValueError('checkpoint/config mismatch')
        policy=policy_config.create_trained_policy(cfg,checkpoint,pytorch_device=args.device)
        rng=np.random.default_rng(args.seed); predictions=[]; targets=[]; holds=[]; latencies=[]
        for idx in indices:
            sample=dataset[idx]
            obs={k:np.asarray(v) if k!='prompt' else v for k,v in sample.items()
                 if k=='prompt' or k.startswith('observation.')}
            before=time.perf_counter()
            action=policy.infer(obs,noise=rng.standard_normal((cfg.model.action_horizon,32),dtype=np.float32))['actions']
            if not np.isfinite(action).all(): raise ValueError('nonfinite predictions')
            predictions.append(action);targets.append(np.asarray(sample['action']))
            holds.append(np.tile(np.asarray(sample['observation.state']),(cfg.model.action_horizon,1)))
            latencies.append(time.perf_counter()-before)
        pred=np.asarray(predictions);target=np.asarray(targets)
        report={'checkpoint':str(checkpoint),'samples':len(indices),'seed':args.seed,
                'mae_by_group':metrics(pred,target),'hold_baseline_mae':metrics(np.asarray(holds),target),
                'latency_s':latencies,'motion_enabled':False,'not_task_success_rate':True}
        np.savez_compressed(args.output/f'{checkpoint.name}.npz',predictions=pred,targets=target)
        (args.output/f'{checkpoint.name}.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report['mae_by_group']),flush=True)
        del policy
        import gc,torch
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=128);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.samples<1:p.error('samples must be positive')
    main(a)
