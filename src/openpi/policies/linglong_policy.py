"""Linglong end-effector SFT contract (no robot commands).

State and action are both 14D, then padded to the pi0.5 width of 32.
0:3 left xyz (m), 3:6 left rpy (rad), 6 left gripper,
7:10 right xyz (m), 10:13 right rpy (rad), 13 right gripper.
All 14 dimensions are supervised. Expected orientations must already be radians.
"""
import dataclasses
import numpy as np

PROMPT = "Use the right arm to pick up the three round parts from the table one by one and place them on the green conveyor belt."
LAYOUT = "left_xyz3,left_rpy3,left_gripper,right_xyz3,right_rpy3,right_gripper"
STATE_DIM = 14
MODEL_ACTION_DIM = 32

@dataclasses.dataclass(frozen=True)
class LinglongInputs:
    def __call__(self, data):
        state = np.asarray(data['observation.state'], dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f'Linglong state must be finite ({STATE_DIM},)')
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
               'action_dim_mask':np.array([True]*STATE_DIM+[False]*(MODEL_ACTION_DIM-STATE_DIM)),
               'prompt':data.get('prompt',PROMPT)}
        if 'actions' in data:
            actions=np.asarray(data['actions'],dtype=np.float32)
            if actions.ndim!=2 or actions.shape[-1]!=STATE_DIM or not np.isfinite(actions).all():
                raise ValueError(f'Linglong actions must be finite (H,{STATE_DIM})')
            out['actions']=actions
        return out

@dataclasses.dataclass(frozen=True)
class LinglongOutputs:
    def __call__(self,data):
        x=np.asarray(data['actions'])
        if x.ndim!=2 or x.shape[-1]!=32 or not np.isfinite(x).all(): raise ValueError('expected (H,32)')
        return {'actions':x[:,:STATE_DIM].copy()}
