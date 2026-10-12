"""Isolated Linglong SFT registration; existing robot configs are untouched."""
import dataclasses
from pathlib import Path
from openpi.policies import linglong_policy as policy
from openpi import transforms

def get_linglong_configs():
    from openpi.training import config as c
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import optimizer, weight_loaders

    @dataclasses.dataclass(frozen=True)
    class LinglongData(c.DataConfigFactory):
        dataset_root: str = str(Path(__file__).resolve().parents[4]/'data/linglong_0714_sft_15hz')
        def create(self, assets_dirs, model_config):
            root=Path(self.dataset_root)
            base=self.create_base_config(assets_dirs,model_config)
            keys=['observation.state',*[f'observation.images.{k}' for k in ('head','left','right')],'prompt']
            return dataclasses.replace(base,dataset_root=str(root),episode_indices_path=str(root/'train_episodes.json'),
                prompt_from_task=True,action_sequence_keys=('action',),
                repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({**{k:k for k in keys},'actions':'action'})]),
                data_transforms=transforms.Group(inputs=[policy.LinglongInputs()],outputs=[policy.LinglongOutputs()]),
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
        policy_metadata={'layout':policy.LAYOUT,'prompt':policy.PROMPT,'motion_enabled':False,
                         'action_semantics':'absolute ee pose xyz+rpy rad and gripper','fps':15})
    # Separate task and assets; keep 0714/DROID experiments unchanged.
    new_root=root/'data/linglong_0923_sft_train_15hz'
    pick_place=dataclasses.replace(original,name='pi05_linglong_0923_sft',
        data=LinglongData(repo_id='linglong_0923_train',dataset_root=str(new_root),
            assets=c.AssetsConfig(assets_dir=str(new_root/'assets'),asset_id='linglong_h')),
        policy_metadata={**original.policy_metadata,
            'prompt':'Use the right arm to pick up the blue block from the table and place it in the box.'})
    return [original,pick_place]
