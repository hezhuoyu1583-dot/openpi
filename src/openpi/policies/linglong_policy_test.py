import numpy as np
import pytest
from openpi.policies.linglong_policy import LinglongInputs, LinglongOutputs, PROMPT

def sample():
    return {'observation.state':np.arange(14,dtype=np.float32),'actions':np.ones((50,14),np.float32),
            **{f'observation.images.{k}':np.full((3,12,16),v,np.float32)
               for k,v in [('head',0.),('left',.5),('right',1.)]}}

def test_training_contract():
    x=LinglongInputs()(sample())
    assert x['prompt']==PROMPT
    np.testing.assert_array_equal(x['state'],np.arange(14))
    assert x['image']['left_wrist_0_rgb'].shape==(12,16,3)
    assert x['image']['left_wrist_0_rgb'][0,0,0]==128
    assert x['action_dim_mask'].tolist()==[True]*14+[False]*18
    assert x['actions'].shape==(50,14)

def test_invalid_values():
    for key in ('observation.state','actions','observation.images.head'):
        s=sample(); s[key].flat[0]=np.nan
        with pytest.raises(ValueError): LinglongInputs()(s)

def test_outputs_preserve_values():
    x=np.arange(1600).reshape(50,32)
    np.testing.assert_array_equal(LinglongOutputs()({'actions':x})['actions'],x[:,:14])
    with pytest.raises(ValueError): LinglongOutputs()({'actions':np.zeros((50,8))})
