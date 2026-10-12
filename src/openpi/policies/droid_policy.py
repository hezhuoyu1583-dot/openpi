import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.training.droid_rlds_dataset import DroidActionSpace


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidLeRobotInputs(transforms.DataTransformFn):
    """Adapt DreamZero/LeRobot DROID-v2.0 records to the OpenPI DROID schema.

    The LeRobot record stores ``observation.state`` as
    ``[cartesian_position(6), gripper_position(1), joint_position(7)]`` and
    ``action`` as ``[cartesian_position(6), cartesian_velocity(6),
    gripper_position(1), gripper_velocity(1), joint_position(7),
    joint_velocity(7)]``.  OpenPI's DROID policy consumes an 8-D vector
    ``[joint_*, gripper_position]``.  This transform performs that projection
    without changing the source dataset.
    """

    model_type: _model.ModelType
    action_space: DroidActionSpace

    def __call__(self, data: dict) -> dict:
        state_raw = np.asarray(data["observation/state"])
        if state_raw.shape[-1] != 14:
            raise ValueError(f"Expected DROID observation.state with 14 dims, got {state_raw.shape}")
        # State layout: cartesian(6), gripper(1), joints(7).
        state = np.concatenate([state_raw[..., 7:14], state_raw[..., 6:7]], axis=-1)

        action_raw = np.asarray(data["action"])
        if action_raw.shape[-1] != 28:
            raise ValueError(f"Expected DROID action with 28 dims, got {action_raw.shape}")
        if self.action_space == DroidActionSpace.JOINT_POSITION:
            arm_action = action_raw[..., 14:21]
        elif self.action_space == DroidActionSpace.JOINT_VELOCITY:
            arm_action = action_raw[..., 21:28]
        else:
            raise ValueError(f"Unsupported DROID action space: {self.action_space}")
        actions = np.concatenate([arm_action, action_raw[..., 12:13]], axis=-1)

        def _parse_image(image) -> np.ndarray:
            image = np.asarray(image)
            if np.issubdtype(image.dtype, np.floating):
                image = (255 * image).astype(np.uint8)
            if image.shape[0] == 3:
                image = einops.rearrange(image, "c h w -> h w c")
            return image

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "actions": actions,
        }
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][..., :8])}
