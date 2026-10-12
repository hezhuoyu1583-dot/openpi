import numpy as np
import pytest

from examples.droid.serve_pi05_roboarena import Pi05DroidRoboArenaPolicy


def _observation():
    return {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "pick up the object",
    }


class FakePolicy:
    def __init__(self, actions):
        self.actions = actions

    def infer(self, obs):
        return {"actions": self.actions, "policy_timing": {"infer_ms": 0.1}}


def test_roboarena_adapter_returns_receding_horizon_prefix():
    actions = np.zeros((15, 8), dtype=np.float32)
    actions[:, 0] = np.arange(15)
    actions[:, 7] = np.array([0.0, 0.6] * 7 + [0.0], dtype=np.float32)
    result = Pi05DroidRoboArenaPolicy(FakePolicy(actions), execution_horizon=8).infer(_observation())
    assert result["actions"].shape == (8, 8)
    expected = actions[:8].copy()
    expected[:, 0] = np.clip(expected[:, 0], -1, 1)
    expected[:, 7] = (expected[:, 7] > 0.5).astype(np.float32)
    np.testing.assert_array_equal(result["actions"], expected)


def test_roboarena_adapter_clips_arm_and_binarizes_gripper():
    actions = np.zeros((15, 8), dtype=np.float32)
    actions[0, :7] = [-2.0, -1.0, -0.2, 0.0, 0.2, 1.0, 2.0]
    actions[0, 7] = 0.75
    result = Pi05DroidRoboArenaPolicy(FakePolicy(actions), execution_horizon=8).infer(_observation())
    np.testing.assert_allclose(result["actions"][0], [ -1.0, -1.0, -0.2, 0.0, 0.2, 1.0, 1.0, 1.0])


def test_roboarena_adapter_rejects_bad_model_output():
    with pytest.raises(ValueError, match="action shape"):
        Pi05DroidRoboArenaPolicy(FakePolicy(np.zeros((10, 8), dtype=np.float32))).infer(_observation())


def test_roboarena_adapter_rejects_bad_input_shape():
    obs = _observation()
    obs["observation/joint_position"] = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError, match="joint_position"):
        Pi05DroidRoboArenaPolicy(FakePolicy(np.zeros((15, 8), dtype=np.float32))).infer(obs)
