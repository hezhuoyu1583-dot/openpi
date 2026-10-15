import numpy as np
import pytest
from openpi.policies.linglong_policy import (
    EEF_CONTROL_MODE, LinglongAbsoluteActions, LinglongInputs, LinglongOutputs, LinglongRelativeActions,
    PROMPT, joint_relative_mask, pose_to_absolute, pose_to_relative, resolve_encoding, with_eef_control_mode)

def images():
    return {f'observation.images.{k}':np.full((3,12,16),v,np.float32)
            for k,v in [('head',0.),('left',.5),('right',1.)]}

def sample(state_dim, action_dim):
    return {'observation.state':np.arange(state_dim,dtype=np.float32),
            'actions':np.ones((50,action_dim),np.float32), **images()}

def test_eef_contract():
    x=LinglongInputs(state_eef=True, action_eef=True)(sample(16,16))
    assert x['prompt']==f'{EEF_CONTROL_MODE} {PROMPT}'
    np.testing.assert_array_equal(x['state'],np.arange(16))
    assert x['image']['left_wrist_0_rgb'].shape==(12,16,3)
    assert x['image']['left_wrist_0_rgb'][0,0,0]==128
    assert x['action_dim_mask'].tolist()==[True]*16+[False]*16
    assert x['actions'].shape==(50,16)

def test_joint_contract():
    x=LinglongInputs()(sample(24,24))
    assert x['prompt']==PROMPT
    assert EEF_CONTROL_MODE not in x['prompt']
    assert x['state'].shape==(24,)
    assert x['action_dim_mask'].tolist()==[True]*22+[False]*10
    assert x['actions'].shape==(50,24)

def test_joint_state_with_eef_action():
    x=LinglongInputs(state_eef=False, action_eef=True)(sample(24,16))
    assert x['prompt'].startswith(EEF_CONTROL_MODE)
    assert x['state'].shape==(24,)
    assert x['actions'].shape==(50,16)
    assert x['action_dim_mask'].sum()==16

def test_control_mode_only_on_eef_actions():
    joint_state=LinglongInputs(state_eef=True, action_eef=False)(sample(16,24))
    assert joint_state['prompt']==PROMPT
    tagged=LinglongInputs(action_eef=True)({**sample(24,16),'prompt':f'{EEF_CONTROL_MODE} {PROMPT}'})
    assert tagged['prompt']==f'{EEF_CONTROL_MODE} {PROMPT}'
    assert with_eef_control_mode(f'  {EEF_CONTROL_MODE} already')==f'{EEF_CONTROL_MODE} already'

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

def test_joint_relative_roundtrip_keeps_gripper_and_base():
    mask = joint_relative_mask()
    assert mask.tolist() == [True]*7+[False]+[True]*7+[False]+[True]*6+[False]*2
    state = np.linspace(0.1, 2.4, 24)
    actions = np.stack([state, state+0.05])
    actions[:, 7] = [0.2, 0.8]
    actions[:, 15] = [0.3, 0.1]
    actions[:, 22:] = [[1, 2], [3, 4]]
    relative = pose_to_relative(state, actions, action_eef=False)
    np.testing.assert_allclose(relative[:, :7], actions[:, :7]-state[:7])
    np.testing.assert_allclose(relative[:, 7], actions[:, 7])
    np.testing.assert_allclose(relative[:, 15], actions[:, 15])
    np.testing.assert_allclose(relative[:, 22:], actions[:, 22:])
    restored = LinglongAbsoluteActions()({'state': state, 'actions': relative})['actions']
    np.testing.assert_allclose(restored, actions, atol=1e-6)
    untouched = LinglongRelativeActions()({'state': state})
    assert 'actions' not in untouched

def test_eef_relative_pose_roundtrip():
    state = np.zeros(16)
    state[0:3] = [0.30, 0.25, 0.65]
    state[3:7] = [1, 0, 0, 0]
    state[7] = 0.25
    yaw = np.pi/3
    state[8:11] = [0.40, -0.10, 0.55]
    state[11:15] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    state[15] = 0.75
    actions = np.repeat(state[None], 2, axis=0)
    actions[1, 0:3] += [0.02, -0.01, 0.03]
    actions[1, 7] = 0
    actions[1, 15] = 1
    relative = LinglongRelativeActions(action_eef=True)({'state': state, 'actions': actions})['actions']
    np.testing.assert_allclose(relative[:, 7], actions[:, 7])
    np.testing.assert_allclose(relative[:, 15], actions[:, 15])
    np.testing.assert_allclose(relative[0, 0:3], 0, atol=1e-6)
    np.testing.assert_allclose(relative[0, 3:7], [1, 0, 0, 0], atol=1e-6)
    restored = pose_to_absolute(state, relative, action_eef=True)
    np.testing.assert_allclose(restored[:, :3], actions[:, :3], atol=1e-6)
    np.testing.assert_allclose(restored[:, 8:11], actions[:, 8:11], atol=1e-6)
    np.testing.assert_allclose(restored[:, [7, 15]], actions[:, [7, 15]])
    for row in range(2):
        for sl in (slice(3, 7), slice(11, 15)):
            assert abs(np.dot(restored[row, sl], actions[row, sl])) > 1-1e-6

def test_outputs_preserve_values():
    x=np.arange(1600).reshape(50,32)
    np.testing.assert_array_equal(LinglongOutputs(action_eef=True)({'actions':x})['actions'],x[:,:16])
    np.testing.assert_array_equal(LinglongOutputs()({'actions':x})['actions'],x[:,:24])
    with pytest.raises(ValueError): LinglongOutputs()({'actions':np.zeros((50,8))})
