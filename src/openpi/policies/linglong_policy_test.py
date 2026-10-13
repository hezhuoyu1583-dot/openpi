import numpy as np
import pytest
from openpi.policies.linglong_policy import LinglongInputs, LinglongOutputs, PROMPT, resolve_encoding

def images():
    return {f'observation.images.{k}':np.full((3,12,16),v,np.float32)
            for k,v in [('head',0.),('left',.5),('right',1.)]}

def sample(state_dim, action_dim):
    return {'observation.state':np.arange(state_dim,dtype=np.float32),
            'actions':np.ones((50,action_dim),np.float32), **images()}

def test_eef_contract():
    x=LinglongInputs(state_eef=True, action_eef=True)(sample(16,16))
    assert x['prompt']==PROMPT
    np.testing.assert_array_equal(x['state'],np.arange(16))
    assert x['image']['left_wrist_0_rgb'].shape==(12,16,3)
    assert x['image']['left_wrist_0_rgb'][0,0,0]==128
    assert x['action_dim_mask'].tolist()==[True]*16+[False]*16
    assert x['actions'].shape==(50,16)

def test_joint_contract():
    x=LinglongInputs()(sample(24,24))
    assert x['state'].shape==(24,)
    assert x['action_dim_mask'].tolist()==[True]*22+[False]*10
    assert x['actions'].shape==(50,24)

def test_joint_state_with_eef_action():
    x=LinglongInputs(state_eef=False, action_eef=True)(sample(24,16))
    assert x['state'].shape==(24,)
    assert x['actions'].shape==(50,16)
    assert x['action_dim_mask'].sum()==16

def test_encoding_flag():
    assert resolve_encoding(True) == (True, True)
    assert resolve_encoding(False) == (False, False)
    assert resolve_encoding(True, 'joint') == (False, True)
    assert resolve_encoding(False, 'eef') == (True, False)
    with pytest.raises(ValueError): resolve_encoding(True, 'xyz')

def test_invalid_values():
    transform=LinglongInputs(state_eef=True, action_eef=True)
    for key in ('observation.state','actions','observation.images.head'):
        s=sample(16,16); s[key].flat[0]=np.nan
        with pytest.raises(ValueError): transform(s)
    with pytest.raises(ValueError): LinglongInputs()(sample(16,16))

def test_outputs_preserve_values():
    x=np.arange(1600).reshape(50,32)
    np.testing.assert_array_equal(LinglongOutputs(action_eef=True)({'actions':x})['actions'],x[:,:16])
    np.testing.assert_array_equal(LinglongOutputs()({'actions':x})['actions'],x[:,:24])
    with pytest.raises(ValueError): LinglongOutputs()({'actions':np.zeros((50,8))})
