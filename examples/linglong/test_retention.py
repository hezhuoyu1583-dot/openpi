import importlib.util
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import tempfile

spec=importlib.util.spec_from_file_location('linglong_train',Path(__file__).with_name('train.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@dataclasses.dataclass
class Config:
    checkpoint_dir:Path
    name:str='pi05_linglong_sft'
    save_interval:int=500
    num_train_steps:int=5000
    max_checkpoints:int|None=None
    model:object=None
    policy_metadata:object=None
    wandb_enabled:bool=False

def test_all_weights_and_two_optimizers():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);cfg=Config(root,model=SimpleNamespace(action_horizon=25))
        def original(model,optimizer,step,config,is_main,data):
            path=root/str(step);path.mkdir()
            for name in ('model.safetensors','metadata.pt','optimizer.pt'):(path/name).write_text('test')
        trainer=SimpleNamespace(save_checkpoint=original);module.install_hook(trainer)
        for step in (1,499,500,1000,1500,4999,5000):trainer.save_checkpoint(None,None,step,cfg,True,None)
        assert sorted(p.name for p in root.iterdir())==['1000','1500','500','5000']
        assert all((root/str(step)/'model.safetensors').exists() for step in (500,1000,1500,5000))
        assert sorted(p.parent.name for p in root.glob('*/optimizer.pt'))==['1500','5000']

def test_real_saver_and_restore_files():
    import torch
    import safetensors.torch
    trainer_spec=importlib.util.spec_from_file_location('retention_real_trainer',Path(__file__).resolve().parents[2]/'scripts/train_pytorch.py')
    trainer=importlib.util.module_from_spec(trainer_spec);trainer_spec.loader.exec_module(trainer)
    module.install_hook(trainer)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);cfg=Config(root,num_train_steps=1601,model=SimpleNamespace(action_horizon=25))
        model=torch.nn.Linear(2,1); opt=torch.optim.AdamW(model.parameters())
        model(torch.ones(1,2)).sum().backward();opt.step()
        data=SimpleNamespace(norm_stats=None,asset_id=None)
        for step in (500,1000,1500,1600,1601): trainer.save_checkpoint(model,opt,step,cfg,True,data)
        assert sorted(int(p.name) for p in root.iterdir())==[500,1000,1500,1601]
        assert sorted(int(p.parent.name) for p in root.glob('*/optimizer.pt'))==[1500,1601]
        restored=torch.nn.Linear(2,1)
        safetensors.torch.load_model(restored,root/'1601/model.safetensors')
        optimizer=torch.optim.AdamW(restored.parameters())
        optimizer.load_state_dict(torch.load(root/'1601/optimizer.pt',weights_only=False))
        torch.testing.assert_close(restored.weight,model.weight)
        assert optimizer.state_dict()['state']
        meta=torch.load(root/'1601/metadata.pt',weights_only=False)
        assert meta['global_step']==1601 and meta['config']['num_train_steps']==1601
