import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import tianyi_policy


@pytest.mark.parametrize(
    ("embodiment", "data_dim"),
    [
        (tianyi_policy.TianyiEmbodiment.STATIONARY, 16),
        (tianyi_policy.TianyiEmbodiment.MOBILE, 25),
    ],
)
def test_tianyi_inputs_support_both_embodiments(embodiment, data_dim):
    transform = tianyi_policy.TianyiInputs(model_type=_model.ModelType.PI05, embodiment=embodiment)
    image = np.ones((3, 32, 48), dtype=np.float32)
    actions = np.arange(4 * data_dim, dtype=np.float32).reshape(4, data_dim)

    result = transform(
        {
            "observation/image": image,
            "observation/state": np.arange(data_dim, dtype=np.float32),
            "actions": actions,
            "prompt": b"move the object",
        }
    )

    assert result["state"].shape == (data_dim,)
    assert result["actions"].shape == (4, data_dim)
    np.testing.assert_array_equal(
        result["action_dim_mask"],
        np.asarray([True] * data_dim + [False] * (32 - data_dim)),
    )
    assert result["image"]["base_0_rgb"].shape == (32, 48, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    }
    assert result["prompt"] == "move the object"


def test_tianyi_layout_masks_are_embodiment_specific():
    stationary = tianyi_policy.get_layout("tianyi")
    mobile = tianyi_policy.get_layout("tianyi_mobile")

    assert stationary.valid_action_mask(32) == (True,) * 16 + (False,) * 16
    assert mobile.valid_action_mask(32) == (True,) * 25 + (False,) * 7
    assert stationary.arm_delta_mask() == (True,) * 14 + (False,) * 2
    assert mobile.arm_delta_mask() == (True,) * 14 + (False,) * 11


@pytest.mark.parametrize(("embodiment", "data_dim"), [("tianyi", 16), ("tianyi_mobile", 25)])
def test_tianyi_outputs_remove_model_padding(embodiment, data_dim):
    actions = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
    result = tianyi_policy.TianyiOutputs(embodiment=embodiment)({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :data_dim])


def test_tianyi_inputs_reject_cross_embodiment_shape():
    transform = tianyi_policy.TianyiInputs(
        model_type=_model.ModelType.PI05,
        embodiment=tianyi_policy.TianyiEmbodiment.MOBILE,
    )

    with pytest.raises(ValueError, match="dimension 25"):
        transform(
            {
                "observation/image": np.zeros((32, 48, 3), dtype=np.uint8),
                "observation/state": np.zeros(16, dtype=np.float32),
            }
        )


def test_tianyi_inputs_reject_model_action_dim_smaller_than_embodiment():
    transform = tianyi_policy.TianyiInputs(
        model_type=_model.ModelType.PI05,
        embodiment=tianyi_policy.TianyiEmbodiment.MOBILE,
        model_action_dim=16,
    )

    with pytest.raises(ValueError, match="smaller than Tianyi data dimension 25"):
        transform(
            {
                "observation/image": np.zeros((32, 48, 3), dtype=np.uint8),
                "observation/state": np.zeros(25, dtype=np.float32),
            }
        )
