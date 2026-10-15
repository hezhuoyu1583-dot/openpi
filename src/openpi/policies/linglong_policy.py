"""Linglong SFT contract (no robot commands).

The proprioception written into the pi0.5 prompt is selected separately from
the supervised action. Joint state/action is 24D (left7, gripper, right7,
gripper, waist4, head2, base2); the base pair is not supervised. EEF
state/action is 16D: xyz, quaternion wxyz, gripper, for each arm. Orientation
is wrist_yaw_link. Both are padded to the pi0.5 width of 32.

Absolute end-effector actions use the pi0.5 control-mode prefix. Joint actions
stay unmarked, matching the released base checkpoint.

Optional relative actions change only the training target: each future pose is
expressed against the current state. Grippers stay absolute. The policy output
transform inverts that change, so evaluation and the robot still receive
absolute joint positions or absolute end-effector poses.
"""
import dataclasses
import numpy as np
from openpi import transforms as _transforms

PROMPT = "Use the right arm to pick up the three round parts from the table one by one and place them on the green conveyor belt."
JOINT_LAYOUT = "left7,left_gripper,right7,right_gripper,waist4,head2,base2"
EEF_LAYOUT = "left_xyz3,left_quat_wxyz4,left_gripper,right_xyz3,right_quat_wxyz4,right_gripper"
LAYOUT = EEF_LAYOUT
JOINT_DIM = 24
JOINT_SUPERVISED = 22
EEF_DIM = 16
STATE_DIM = EEF_DIM
MODEL_ACTION_DIM = 32
# Official absolute end-effector marker. Joint position actions have no marker.
EEF_CONTROL_MODE = "<control_mode> end effector </control_mode>"

def with_eef_control_mode(prompt):
    text = prompt.strip()
    if text.startswith(EEF_CONTROL_MODE):
        return text
    return f"{EEF_CONTROL_MODE} {text}"

def joint_relative_mask():
    """Arms, waist, and head are deltas. Grippers and the base pair stay absolute."""
    return np.asarray(_transforms.make_bool_mask(7, -1, 7, -1, 6, -2), dtype=bool)

def _expand_state(state, actions):
    state = np.asarray(state)
    while state.ndim < np.asarray(actions).ndim:
        state = np.expand_dims(state, axis=-2)
    return state

def _quat_multiply(left, right):
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), axis=-1)

def _quat_conjugate(quat):
    out = np.array(quat, copy=True)
    out[..., 1:] *= -1
    return out

def _normalize_quat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1e-8)

def pose_to_relative(state, actions, *, action_eef):
    """Return actions relative to state. Joint deltas subtract; EEF orientation uses q_state^{-1} ⊗ q_action."""
    state = _expand_state(np.asarray(state, dtype=np.float64), actions)
    actions = np.asarray(actions, dtype=np.float64)
    width = EEF_DIM if action_eef else JOINT_DIM
    out = np.array(actions, copy=True)
    if action_eef:
        for origin in (0, 8):
            out[..., origin:origin + 3] = actions[..., origin:origin + 3] - state[..., origin:origin + 3]
            relative = _quat_multiply(_quat_conjugate(state[..., origin + 3:origin + 7]), actions[..., origin + 3:origin + 7])
            out[..., origin + 3:origin + 7] = _normalize_quat(relative)
        return out
    mask = joint_relative_mask()
    out[..., :width] -= np.where(mask, state[..., :width], 0)
    return out

def pose_to_absolute(state, actions, *, action_eef):
    """Invert pose_to_relative. Grippers and, for joints, the base pair are copied through."""
    state = _expand_state(np.asarray(state, dtype=np.float64), actions)
    actions = np.asarray(actions, dtype=np.float64)
    width = EEF_DIM if action_eef else JOINT_DIM
    out = np.array(actions, copy=True)
    if action_eef:
        for origin in (0, 8):
            out[..., origin:origin + 3] = actions[..., origin:origin + 3] + state[..., origin:origin + 3]
            absolute = _quat_multiply(state[..., origin + 3:origin + 7], _normalize_quat(actions[..., origin + 3:origin + 7]))
            out[..., origin + 3:origin + 7] = _normalize_quat(absolute)
        return out
    mask = joint_relative_mask()
    out[..., :width] += np.where(mask, state[..., :width], 0)
    return out

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
        prompt = data.get('prompt', PROMPT)
        if self.action_eef:
            if isinstance(prompt, bytes):
                prompt = prompt.decode('utf-8')
            elif not isinstance(prompt, str):
                prompt = prompt.item()
            prompt = with_eef_control_mode(prompt)
        out = {'state':state,'image':images,'image_mask':{k:np.True_ for k in images},
               'action_dim_mask':np.array([True]*supervised+[False]*(MODEL_ACTION_DIM-supervised)),
               'prompt':prompt}
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

@dataclasses.dataclass(frozen=True)
class LinglongRelativeActions:
    """Training input: absolute dataset actions become poses relative to the current state."""
    action_eef: bool = False

    def __call__(self, data):
        if 'actions' not in data:
            return data
        data['actions'] = pose_to_relative(data['state'], data['actions'], action_eef=self.action_eef).astype(np.float32)
        return data

@dataclasses.dataclass(frozen=True)
class LinglongAbsoluteActions:
    """Policy output: model deltas become the absolute pose the eval and robot paths already consume."""
    action_eef: bool = False

    def __call__(self, data):
        if 'actions' not in data:
            return data
        width = EEF_DIM if self.action_eef else JOINT_DIM
        actions = np.array(data['actions'], copy=True)
        actions[..., :width] = pose_to_absolute(np.asarray(data['state'])[..., :width], actions[..., :width], action_eef=self.action_eef)
        data['actions'] = actions
        return data
