"""Serve the official pi0.5-DROID checkpoint through the RoboArena protocol.

The standard OpenPI websocket server and RoboArena's policy server use
different request envelopes. This adapter preserves pi05_droid's inputs and
joint-velocity actions while exposing the latter protocol for real-world
RoboArena evaluation.
"""

import logging

import numpy as np
import tyro

from openpi.policies import policy as openpi_policy
from openpi.policies import policy_config
from openpi.training import config as training_config
from roboarena.policy import BasePolicy
from roboarena.policy_server import PolicyServerConfig
from roboarena.policy_server import WebsocketPolicyServer
from roboarena.policy_server import load_security


REQUIRED_OBSERVATION_KEYS = (
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/gripper_position",
    "prompt",
)
IMAGE_KEYS = REQUIRED_OBSERVATION_KEYS[:2]
MODEL_ACTION_HORIZON = 15
ACTION_DIM = 8
GRIPPER_THRESHOLD = 0.5


class Pi05DroidRoboArenaPolicy(BasePolicy):
    """Adapts the stateless OpenPI pi05_droid policy to RoboArena."""

    def __init__(self, policy: openpi_policy.Policy, execution_horizon: int = 8) -> None:
        if not 1 <= execution_horizon <= MODEL_ACTION_HORIZON:
            raise ValueError("execution_horizon must be between 1 and 15")
        self._policy = policy
        self._execution_horizon = execution_horizon

    def infer(self, obs: dict) -> dict:
        self._validate_observation(obs)
        result = self._policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        expected_shape = (MODEL_ACTION_HORIZON, ACTION_DIM)
        if actions.shape != expected_shape:
            raise ValueError(
                "pi05_droid must return action shape "
                f"{expected_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("pi05_droid returned NaN or Inf actions")

        # Keep the RoboArena path identical to the official DROID client.  The
        # client clips each arm component before RobotEnv.step() and converts
        # the predicted gripper position to a binary position command.  The
        # RoboArena evaluator executes every returned row directly, so doing
        # this here is required for both the range contract and parity with
        # openpi/examples/droid/main.py and the MolmoSpaces adapter.
        actions = actions.copy()
        actions[:, :7] = np.clip(actions[:, :7], -1.0, 1.0)
        actions[:, 7] = (actions[:, 7] > GRIPPER_THRESHOLD).astype(np.float32)
        # RoboArena executes every returned row before requesting another chunk.
        # Return only the receding-horizon prefix used by the DROID client.
        return {"actions": actions[: self._execution_horizon]}

    def reset(self, reset_info: dict) -> None:
        # pi05_droid has no rollout-local visual or action cache to clear.
        del reset_info

    @staticmethod
    def _validate_observation(obs: dict) -> None:
        missing = [key for key in REQUIRED_OBSERVATION_KEYS if key not in obs]
        if missing:
            raise ValueError(f"missing required pi05_droid observation keys: {missing}")
        for key in IMAGE_KEYS:
            image = np.asarray(obs[key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{key} must be uint8 HWC with shape (224, 224, 3), "
                    f"got dtype={image.dtype}, shape={image.shape}"
                )
        for key, expected_shape in (
            ("observation/joint_position", (7,)),
            ("observation/gripper_position", (1,)),
        ):
            state = np.asarray(obs[key])
            if state.shape != expected_shape:
                raise ValueError(f"{key} must have shape {expected_shape}, got {state.shape}")
            if not np.issubdtype(state.dtype, np.number) or not np.isfinite(state).all():
                raise ValueError(f"{key} must contain finite numeric values")
        if not isinstance(obs["prompt"], str):
            raise ValueError("prompt must be a string")


def main(
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_droid",
    host: str = "127.0.0.1",
    port: int = 8000,
    execution_horizon: int = 8,
) -> None:
    """Load pi05_droid and expose it with the RoboArena server contract."""
    load_security(host)  # Fail before allocating GPU memory if security is missing.
    config = training_config.get_config("pi05_droid")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    server_config = PolicyServerConfig(
        image_resolution=(224, 224),
        needs_wrist_camera=True,
        n_external_cameras=1,
        needs_stereo_camera=False,
        needs_session_id=False,
        action_space="joint_velocity",
    )
    logging.info("Serving pi05_droid through RoboArena at %s:%d", host, port)
    WebsocketPolicyServer(
        Pi05DroidRoboArenaPolicy(policy, execution_horizon), server_config, host=host, port=port
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
