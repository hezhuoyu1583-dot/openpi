import dataclasses
import enum

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


class TianyiEmbodiment(enum.Enum):
    STATIONARY = "tianyi"
    MOBILE = "tianyi_mobile"


@dataclasses.dataclass(frozen=True)
class TianyiLayout:
    data_dim: int
    left_arm: slice
    right_arm: slice
    left_gripper: slice
    right_gripper: slice
    chassis_twist: slice | None = None
    head_position: slice | None = None

    def valid_action_mask(self, model_action_dim: int) -> tuple[bool, ...]:
        if model_action_dim < self.data_dim:
            raise ValueError(
                f"Model action dimension {model_action_dim} is smaller than Tianyi data dimension {self.data_dim}."
            )
        return (True,) * self.data_dim + (False,) * (model_action_dim - self.data_dim)

    def arm_delta_mask(self) -> tuple[bool, ...]:
        """Mask for an optional arm-only absolute-to-delta transform."""
        mask = np.zeros(self.data_dim, dtype=bool)
        mask[self.left_arm] = True
        mask[self.right_arm] = True
        return tuple(mask.tolist())


TIANYI_LAYOUTS = {
    TianyiEmbodiment.STATIONARY: TianyiLayout(
        data_dim=16,
        left_arm=slice(0, 7),
        right_arm=slice(7, 14),
        left_gripper=slice(14, 15),
        right_gripper=slice(15, 16),
    ),
    TianyiEmbodiment.MOBILE: TianyiLayout(
        data_dim=25,
        left_arm=slice(0, 7),
        right_arm=slice(7, 14),
        left_gripper=slice(14, 15),
        right_gripper=slice(15, 16),
        chassis_twist=slice(16, 22),
        head_position=slice(22, 25),
    ),
}


def get_layout(embodiment: TianyiEmbodiment | str) -> TianyiLayout:
    if isinstance(embodiment, str):
        try:
            embodiment = TianyiEmbodiment(embodiment)
        except ValueError as exc:
            choices = ", ".join(item.value for item in TianyiEmbodiment)
            raise ValueError(f"Unknown Tianyi embodiment {embodiment!r}; expected one of: {choices}.") from exc
    return TIANYI_LAYOUTS[embodiment]


def make_tianyi_example(embodiment: TianyiEmbodiment | str = TianyiEmbodiment.STATIONARY) -> dict:
    layout = get_layout(embodiment)
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(layout.data_dim).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Tianyi camera image must have three dimensions, got {image.shape}.")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Tianyi camera image must be RGB, got {image.shape}.")
    return image


def _validate_last_dim(value: np.ndarray, expected: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 0 or value.shape[-1] != expected:
        raise ValueError(f"{name} must end with dimension {expected}, got {value.shape}.")
    return value


@dataclasses.dataclass(frozen=True)
class TianyiInputs(transforms.DataTransformFn):
    """Convert either Tianyi embodiment to the pi0/pi0.5 input contract.

    The caller chooses the embodiment explicitly. This prevents a 16D stationary
    sample from being silently interpreted as a 25D mobile sample during future
    mixed-embodiment training.
    """

    model_type: _model.ModelType
    embodiment: TianyiEmbodiment | str = TianyiEmbodiment.STATIONARY
    model_action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"TianyiInputs only supports pi0/pi0.5, got {self.model_type}.")

        layout = get_layout(self.embodiment)
        action_dim_mask = np.asarray(layout.valid_action_mask(self.model_action_dim), dtype=bool)
        state = _validate_last_dim(data["observation/state"], layout.data_dim, "Tianyi state")
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            # Keep this per sample so a future mixed-embodiment loader can use a
            # different loss mask for stationary and mobile trajectories.
            "action_dim_mask": action_dim_mask,
        }

        if "actions" in data:
            inputs["actions"] = _validate_last_dim(data["actions"], layout.data_dim, "Tianyi actions")

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class TianyiOutputs(transforms.DataTransformFn):
    embodiment: TianyiEmbodiment | str = TianyiEmbodiment.STATIONARY

    def __call__(self, data: dict) -> dict:
        layout = get_layout(self.embodiment)
        actions = np.asarray(data["actions"])
        if actions.ndim == 0 or actions.shape[-1] < layout.data_dim:
            raise ValueError(
                f"Model actions must contain at least {layout.data_dim} dimensions, got {actions.shape}."
            )
        return {"actions": actions[..., : layout.data_dim]}
