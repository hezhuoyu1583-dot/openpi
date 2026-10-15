"""Isolated Linglong SFT registration; existing robot configs are untouched."""
import dataclasses
import json
import logging
from pathlib import Path
from typing import Literal
import numpy as np
import pyarrow.parquet as pq
import tyro
from openpi.policies import linglong_policy as policy
from openpi.shared.normalize import NormStats
from openpi import transforms

def _eef_norm_stats(root):
    path=root/'assets/linglong_h/eef_norm_stats.json'
    if not path.is_file():
        raise ValueError(f'{path} is required when state or action uses the EEF encoding')
    payload=json.loads(path.read_text())['norm_stats']
    stats={}
    for source,target in (('state_eef','state'),('action_eef','actions')):
        row=payload[source]
        item=NormStats(mean=np.asarray(row['mean'],np.float64),std=np.asarray(row['std'],np.float64),
                       q01=np.asarray(row['q01'],np.float64),q99=np.asarray(row['q99'],np.float64))
        if tuple(item.mean.shape)!=(policy.EEF_DIM,) or tuple(item.q01.shape)!=(policy.EEF_DIM,) or tuple(item.q99.shape)!=(policy.EEF_DIM,):
            raise ValueError(f'{source} norm stats must have shape ({policy.EEF_DIM},)')
        stats[target]=item
    return stats

def _expect_dim(stats, dim, label):
    if stats is None or 'state' not in stats or 'actions' not in stats:
        raise ValueError(f'{label} norm stats are missing')
    for key in ('state','actions'):
        item=stats[key]
        if tuple(np.shape(item.mean))!=(dim,) or tuple(np.shape(item.q01))!=(dim,) or tuple(np.shape(item.q99))!=(dim,):
            raise ValueError(f'{label} {key} norm stats must have shape ({dim},)')

def select_norm_stats(root, joint_stats, state_eef, action_eef):
    eef_stats=_eef_norm_stats(root) if state_eef or action_eef else None
    if not state_eef or not action_eef:
        _expect_dim(joint_stats, policy.JOINT_DIM, 'joint')
    return {'state':(eef_stats if state_eef else joint_stats)['state'],
            'actions':(eef_stats if action_eef else joint_stats)['actions']}

def encoding_metadata(state_eef, action_eef, relative_actions=False):
    absolute = 'absolute ee pose xyz + quaternion wxyz + gripper' if action_eef else 'absolute q_exp'
    return {'state_key':'state_eef' if state_eef else 'observation.state',
            'action_key':'action_eef' if action_eef else 'action',
            'state_encoding':'eef' if state_eef else 'joint',
            'layout':policy.EEF_LAYOUT if action_eef else policy.JOINT_LAYOUT,
            'action_semantics':absolute,
            'relative_actions':bool(relative_actions),
            'model_action_semantics':('relative to the input state; grippers stay absolute'
                                      if relative_actions else absolute),
            'control_mode':policy.EEF_CONTROL_MODE if action_eef else None}

def sync_policy_metadata(config):
    """Record the launch-selected encodings. Omitted state encoding follows the action."""
    data=config.data
    if not hasattr(data,'eef'): return config
    state_eef,action_eef=policy.resolve_encoding(data.eef, data.state_encoding)
    return dataclasses.replace(config, policy_metadata={**config.policy_metadata, **encoding_metadata(state_eef, action_eef, getattr(data,'relative_actions',False)),
        'dataset_root':str(Path(data.dataset_root))})

def _quantile_stats(values):
    values=np.asarray(values, dtype=np.float64)
    return NormStats(mean=values.mean(0), std=values.std(0),
                     q01=np.quantile(values, 0.01, axis=0), q99=np.quantile(values, 0.99, axis=0))

def relative_action_norm_stats(root, episode_indices, action_eef, horizon):
    """Quantile stats of the relative training target, including LeRobot's end-of-episode repeat."""
    if horizon < 1: raise ValueError('action horizon must be positive')
    state_key='state_eef' if action_eef else 'observation.state'
    action_key='action_eef' if action_eef else 'action'
    width=policy.EEF_DIM if action_eef else policy.JOINT_DIM
    chunks=[]
    for episode in episode_indices:
        path=Path(root)/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'
        table=pq.read_table(path, columns=[state_key, action_key])
        state=np.asarray(table.column(state_key).to_pylist(), dtype=np.float64)
        action=np.asarray(table.column(action_key).to_pylist(), dtype=np.float64)
        if state.ndim != 2 or action.shape != state.shape or state.shape[-1] != width:
            raise ValueError(f'{path} {state_key}/{action_key} must have shape (T,{width})')
        length=state.shape[0]
        steps=np.arange(horizon)
        source=np.minimum(np.arange(length)[:,None]+steps, length-1)
        chunks.append(policy.pose_to_relative(state[:,None,:], action[source], action_eef=action_eef).reshape(-1, width))
    if not chunks: raise ValueError('no episodes for relative action stats')
    stats=_quantile_stats(np.concatenate(chunks, axis=0))
    logging.info('Relative action norm stats from %d episodes, horizon %d, frames %d',
                 len(episode_indices), horizon, sum(len(c) for c in chunks))
    return stats

def get_linglong_configs():
    from openpi.training import config as c
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import optimizer, weight_loaders

    @dataclasses.dataclass(frozen=True)
    class LinglongData(c.DataConfigFactory):
        # Episodes and both norm-stats files come from this directory.
        # Override it with --data.dataset-root; the assets directory follows.
        dataset_root: str = str(Path(__file__).resolve().parents[4]/'data/linglong_0714_sft_15hz')
        # Action column and loss mask. The state written into the prompt follows this
        # unless state_encoding is set on the command line.
        eef: bool = False
        state_encoding: Literal['joint', 'eef'] | None = None
        # Train on poses relative to the current state. Policy outputs convert back to absolute.
        relative_actions: bool = False
        # Checkpoint loading already has the saved action stats. Not a launch flag.
        skip_relative_norm_stats: tyro.conf.Suppress[bool] = False
        def create(self, assets_dirs, model_config):
            root=Path(self.dataset_root)
            asset_id=self.assets.asset_id or 'linglong_h'
            aligned=dataclasses.replace(self, assets=dataclasses.replace(
                self.assets, assets_dir=str(root/'assets'), asset_id=asset_id))
            base=aligned.create_base_config(assets_dirs,model_config)
            state_eef,action_eef=policy.resolve_encoding(self.eef, self.state_encoding)
            if self.relative_actions and state_eef != action_eef:
                raise ValueError('relative actions require the state and action encodings to match')
            state_key='state_eef' if state_eef else 'observation.state'
            action_key='action_eef' if action_eef else 'action'
            image_keys=[f'observation.images.{k}' for k in ('head','left','right')]
            norm_stats=select_norm_stats(root, base.norm_stats, state_eef, action_eef)
            if self.relative_actions and not self.skip_relative_norm_stats:
                episodes=json.loads((root/'train_episodes.json').read_text())
                norm_stats={**norm_stats, 'actions':relative_action_norm_stats(
                    root, [int(i) for i in episodes], action_eef, model_config.action_horizon)}
            inputs=[policy.LinglongInputs(state_eef=state_eef, action_eef=action_eef)]
            outputs=[policy.LinglongOutputs(action_eef=action_eef)]
            if self.relative_actions:
                inputs.append(policy.LinglongRelativeActions(action_eef=action_eef))
                outputs.insert(0, policy.LinglongAbsoluteActions(action_eef=action_eef))
            return dataclasses.replace(base,dataset_root=str(root),episode_indices_path=str(root/'train_episodes.json'),
                prompt_from_task=True,action_sequence_keys=(action_key,),
                norm_stats=norm_stats,
                repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                    **{k:k for k in (*image_keys,'prompt')},'observation.state':state_key,'actions':action_key})]),
                data_transforms=transforms.Group(inputs=inputs, outputs=outputs),
                model_transforms=c.ModelTransformFactory(default_prompt=policy.PROMPT)(model_config))

    root=Path(__file__).resolve().parents[4]
    original=c.TrainConfig(name='pi05_linglong_sft',
        model=Pi0Config(pi05=True,action_dim=32,action_horizon=25),
        data=LinglongData(repo_id='linglong_0714',assets=c.AssetsConfig(
            assets_dir=str(root/'data/linglong_0714_sft_15hz/assets'),asset_id='linglong_h')),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(root/'data/checkpoints/pi05_base/params')),
        pytorch_weight_path=str(root/'data/checkpoints/pi05_base_pytorch'),
        checkpoint_base_dir=str(root/'data/checkpoints'),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100,peak_lr=1e-5,decay_steps=5000,decay_lr=1e-6),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.),batch_size=64,num_workers=0,
        num_train_steps=5000,save_interval=500,log_interval=20,max_checkpoints=None,wandb_enabled=False,
        policy_metadata={'layout':policy.JOINT_LAYOUT,'prompt':policy.PROMPT,'motion_enabled':False,
                         'state_key':'observation.state','action_key':'action','state_encoding':'joint',
                         'action_semantics':'absolute q_exp','fps':15})
    # Separate task and assets; keep 0714/DROID experiments unchanged.
    new_root=root/'data/linglong_0923_sft_train_15hz'
    pick_place=dataclasses.replace(original,name='pi05_linglong_0923_sft',
        data=LinglongData(repo_id='linglong_0923_train',dataset_root=str(new_root),
            assets=c.AssetsConfig(assets_dir=str(new_root/'assets'),asset_id='linglong_h')),
        policy_metadata={**original.policy_metadata,
            'prompt':'Use the right arm to pick up the blue block from the table and place it in the box.'})
    eef_root=root/'data/linglong_0923_1010_eef_sft_train_15hz'
    eef=dataclasses.replace(original,name='pi05_linglong_0923_1010_eef_sft',
        data=LinglongData(repo_id='linglong_0923_1010_eef',dataset_root=str(eef_root),eef=True,
            assets=c.AssetsConfig(assets_dir=str(eef_root/'assets'),asset_id='linglong_h')),
        policy_metadata={**original.policy_metadata,'prompt_from_task':True,
            **encoding_metadata(True, True)})
    return [original,pick_place,eef]
