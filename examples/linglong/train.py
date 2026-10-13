"""Linglong-only checkpoint retention hook; shared trainer remains unchanged."""
import dataclasses
import importlib.util
import json
import logging
import os
from pathlib import Path

def retain_optimizer_states(root, keep=2):
    completed=sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()
                      and (p/'model.safetensors').is_file() and (p/'metadata.pt').is_file()),key=lambda p:int(p.name))
    for old in completed[:-keep]:
        (old/'optimizer.pt').unlink(missing_ok=True)

def install_hook(trainer):
    original=trainer.save_checkpoint
    def save(model,optimizer,step,config,is_main,data_config):
        if not is_main or not (step>0 and (step%config.save_interval==0 or step==config.num_train_steps)):
            return
        # Existing trainer saves a non-divisible final step one step early.
        # Correct that predicate only for Linglong; regular saves keep the exact config.
        original(model,optimizer,step,dataclasses.replace(config,max_checkpoints=None) if step%config.save_interval==0
                 else dataclasses.replace(config,num_train_steps=step+1,max_checkpoints=None),is_main,data_config)
        path=config.checkpoint_dir/str(step)
        if step%config.save_interval:
            import torch
            metadata=torch.load(path/'metadata.pt',map_location='cpu',weights_only=False)
            metadata['config']=dataclasses.asdict(config)
            torch.save(metadata,path/'metadata.pt')
        (path/'inference_config.json').write_text(json.dumps({'config_name':config.name,
            'horizon':config.model.action_horizon,'action_dim':32,'policy_metadata':config.policy_metadata,
            'save_interval':config.save_interval,'optimizer_retention':2},indent=2))
        retain_optimizer_states(config.checkpoint_dir)
    trainer.save_checkpoint=save

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('linglong_shared_trainer',root/'scripts/train_pytorch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    install_hook(module)
    module.init_logging()
    config=module._config.cli()
    if config.name not in ('pi05_linglong_sft','pi05_linglong_0923_sft','pi05_linglong_0923_1010_eef_sft'):
        raise ValueError('Linglong-only entry point')
    from openpi.training.misc.linglong_config import sync_policy_metadata
    config=sync_policy_metadata(config)
    # Outside checkpoint_dir: the shared trainer checks its existence on startup.
    handlers=[]
    if int(os.environ.get('RANK','0'))==0:
        from metrics import MetricsHandler, plot
        log_dir=root/'data/training_logs/linglong'/config.exp_name
        log_dir.mkdir(parents=True,exist_ok=True)
        file_handler=logging.FileHandler(log_dir/'train.log',mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        handlers=[file_handler,MetricsHandler(log_dir)]
        for handler in handlers:logging.getLogger().addHandler(handler)
        logging.info('Linglong run start: resume=%s config=%s state=%s action=%s metrics_run_id=%s',
                     config.resume,config.name,config.policy_metadata.get('state_key'),
                     config.policy_metadata.get('action_key'),handlers[1].run_id)
    try:
        module.train_loop(config)
    except BaseException:
        logging.exception('Linglong training exited with an exception')
        raise
    finally:
        for handler in handlers:
            logging.getLogger().removeHandler(handler);handler.close()
        if handlers:
            try: plot(log_dir)
            except Exception: logging.exception('Could not render training curves; metrics remain on disk')
