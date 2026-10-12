import numpy as np

from openpi.models.model import ModelType
from openpi.policies.droid_policy import DroidLeRobotInputs
from openpi.training.droid_rlds_dataset import DroidActionSpace


def _sample():
    return {
        "observation/state": np.arange(14, dtype=np.float32),
        "action": np.arange(2 * 28, dtype=np.float32).reshape(2, 28),
        "observation/exterior_image_1_left": np.zeros((3, 4, 5), dtype=np.float32),
        "observation/exterior_image_2_left": np.zeros((3, 4, 5), dtype=np.float32),
        "observation/wrist_image_left": np.zeros((3, 4, 5), dtype=np.float32),
        "prompt": "pick",
    }


def test_droid_lerobot_velocity_projection():
    output = DroidLeRobotInputs(ModelType.PI05, DroidActionSpace.JOINT_VELOCITY)(_sample())
    np.testing.assert_array_equal(output["state"], np.arange(7, 14).tolist() + [6])
    np.testing.assert_array_equal(output["actions"][0], np.arange(21, 28).tolist() + [12])
    assert output["actions"].shape == (2, 8)


def test_droid_lerobot_position_projection():
    output = DroidLeRobotInputs(ModelType.PI05, DroidActionSpace.JOINT_POSITION)(_sample())
    np.testing.assert_array_equal(output["actions"][0], np.arange(14, 21).tolist() + [12])
