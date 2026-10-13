"""Linglong SFT contract (no robot commands).

The proprioception written into the pi0.5 prompt is selected separately from
the supervised action. Joint state/action is 24D (left7, gripper, right7,
gripper, waist4, head2, base2); the base pair is not supervised. EEF
state/action is 16D: xyz, quaternion wxyz, gripper, for each arm. Orientation
is wrist_yaw_link. Both are padded to the pi0.5 width of 32.
"""
import dataclasses
import numpy as np

PROMPT = "Use the right arm to pick up the three round parts from the table one by one and place them on the green conveyor belt."
JOINT_LAYOUT = "left7,left_gripper,right7,right_gripper,waist4,head2,base2"
EEF_LAYOUT = "left_xyz3,left_quat_wxyz4,left_gripper,right_xyz3,right_quat_wxyz4,right_gripper"
LAYOUT = EEF_LAYOUT
JOINT_DIM = 24
JOINT_SUPERVISED = 22
EEF_DIM = 16
STATE_DIM = EEF_DIM
MODEL_ACTION_DIM = 32

def resolve_encoding(action_eef, state_encoding=None):
    """Return (state_eef, action_eef). An omitted state encoding follows the action."""
    if state_encoding not in (None, "joint", "eef"):
        raise ValueError("state_encoding must be 'joint', 'eef', or omitted")
    action_eef = bool(action_eef)
    return (action_eef if state_encoding is None else state_encoding == "eef"), action_eef

@dataclasses.dataclass(frozen=True)
class LinglongInputs:
    state_eef: bool = False
    action_eef: bool = False

    def __call__(self, data):
        state_dim = EEF_DIM if self.state_eef else JOINT_DIM
        action_dim = EEF_DIM if self.action_eef else JOINT_DIM
        supervised = EEF_DIM if self.action_eef else JOINT_SUPERVISED
        state = np.asarray(data['observation.state'], dtype=np.float32)
        if state.shape != (state_dim,) or not np.isfinite(state).all():
            raise ValueError(f'Linglong state must be finite ({state_dim},)')
        images = {}
        for source, target in zip(('head','left','right'), ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')):
            im = np.asarray(data['observation.images.'+source])
            if im.ndim != 3: raise ValueError('image must have 3 axes')
            if im.shape[0] == 3: im = np.transpose(im,(1,2,0))
            if im.shape[-1] != 3: raise ValueError('image must be RGB')
            if np.issubdtype(im.dtype,np.floating):
                if not np.isfinite(im).all() or im.min()<0 or im.max()>1: raise ValueError('float image must be [0,1]')
                im = np.rint(im*255).astype(np.uint8)
            if im.dtype != np.uint8: raise ValueError('image must be uint8 or float [0,1]')
            images[target] = im
        out = {'state':state,'image':images,'image_mask':{k:np.True_ for k in images},
               'action_dim_mask':np.array([True]*supervised+[False]*(MODEL_ACTION_DIM-supervised)),
               'prompt':data.get('prompt',PROMPT)}
        if 'actions' in data:
            actions=np.asarray(data['actions'],dtype=np.float32)
            if actions.ndim!=2 or actions.shape[-1]!=action_dim or not np.isfinite(actions).all():
                raise ValueError(f'Linglong actions must be finite (H,{action_dim})')
            out['actions']=actions
        return out

@dataclasses.dataclass(frozen=True)
class LinglongOutputs:
    action_eef: bool = False

    def __call__(self,data):
        width = EEF_DIM if self.action_eef else JOINT_DIM
        x=np.asarray(data['actions'])
        if x.ndim!=2 or x.shape[-1]!=MODEL_ACTION_DIM or not np.isfinite(x).all(): raise ValueError('expected (H,32)')
        return {'actions':x[:,:width].copy()}
