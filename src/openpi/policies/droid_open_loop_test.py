import numpy as np

from examples.droid.evaluate_pi05_droid_open_loop import compute_metrics


def test_compute_metrics_zero_error():
    target = np.zeros((2, 15, 8), dtype=np.float32)
    result = compute_metrics(target.copy(), target)
    assert result["arm_mae"] == 0.0
    assert result["overall_rmse"] == 0.0
    assert result["gripper_f1"] == 1.0


def test_compute_metrics_horizon_and_gripper():
    target = np.zeros((1, 15, 8), dtype=np.float32)
    pred = target.copy()
    pred[:, :, 0] = np.arange(15, dtype=np.float32)
    pred[:, :, 7] = 1.0
    result = compute_metrics(pred, target)
    np.testing.assert_allclose(result["horizon_arm_mae"], np.arange(15, dtype=np.float32) / 7)
    assert result["gripper_f1"] == 0.0
    assert len(result["per_joint_mae"]) == 7
