"""Forward kinematics for the 24-D Linglong joint vector.

Copied from linglong-h-robot/tool/eval_stitch_error.py. Link lengths and the
wrist TCP offset are constants; this module does not read an MJCF.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


LEFT_HAND_TCP = np.array([-0.00808116, -0.00004938, 0.01828485], dtype=np.float64)
RIGHT_HAND_TCP = np.array([0.00801431, 0.00016277, 0.01823565], dtype=np.float64)
# Fixed joint wrist_yaw_link -> hand_base_link. Left rpy=(0, pi/2, 0). Right rpy=(0, -pi/2, pi). wxyz.
LEFT_HAND_MOUNT = np.array([0.7071067812, 0.0, 0.7071067812, 0.0], dtype=np.float64)
RIGHT_HAND_MOUNT = np.array([0.0, 0.7071067812, 0.0, 0.7071067812], dtype=np.float64)


def _link(pos: Sequence[float], axis: Sequence[float] | None = None, index: int | None = None,
          quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> tuple:
    return (
        np.asarray(pos, dtype=np.float64),
        np.asarray(quat, dtype=np.float64),
        None if axis is None else np.asarray(axis, dtype=np.float64),
        index,
    )


# Hinge chain from base_link, joint order matching the 24-D training vector.
WAIST_LINKS = (
    _link((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 16),
    _link((0.28, 0.0, 0.105), (0.0, 1.0, 0.0), 17),
    _link((-0.28, 0.0, 0.105), (0.0, 1.0, 0.0), 18),
    _link((0.0, 0.0, 0.139), (0.0, 0.0, 1.0), 19),
)
LEFT_ARM_LINKS = (
    _link((0.0, 0.21175, 0.39195), (0.0, 1.0, 0.0), 0),
    _link((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1),
    _link((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2),
    _link((0.00005, 0.0, -0.26), (0.0, 1.0, 0.0), 3),
    _link((0.198, 0.0, 0.0), (1.0, 0.0, 0.0), 4),
    _link((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 5),
    _link((0.062, 0.0, 0.0), (0.0, 0.0, 1.0), 6),
)
RIGHT_ARM_LINKS = (
    _link((0.0, -0.21175, 0.39195), (0.0, 1.0, 0.0), 8),
    _link((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 9),
    _link((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 10),
    _link((0.0, 0.0, -0.26), (0.0, 1.0, 0.0), 11),
    _link((0.198, 0.0, 0.0), (1.0, 0.0, 0.0), 12),
    _link((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 13),
    _link((0.062, 0.0, 0.0), (0.0, 0.0, 1.0), 14),
)


def _rotation_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _wrist_tcp(mount_quat: np.ndarray, hand_tcp: np.ndarray) -> np.ndarray:
    """TCP in wrist_yaw_link: 63 mm mount translation, then the hand-frame offset."""
    mount = _rotation_from_quat(mount_quat)
    return np.array([0.063, 0.0, 0.0], dtype=np.float64) + mount @ hand_tcp


LEFT_TCP_OFFSET = _wrist_tcp(LEFT_HAND_MOUNT, LEFT_HAND_TCP)
RIGHT_TCP_OFFSET = _wrist_tcp(RIGHT_HAND_MOUNT, RIGHT_HAND_TCP)


def _rotation_from_hinge(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = (float(value) for value in axis)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    complement = 1.0 - cosine
    return np.array(
        [
            [cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine],
            [y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine],
            [z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement],
        ],
        dtype=np.float64,
    )


def _body_pose(joints: np.ndarray, links: Sequence[tuple], rotation: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for pos, quat, axis, index in links:
        position = position + rotation @ pos
        rotation = rotation @ _rotation_from_quat(quat)
        if axis is not None:
            rotation = rotation @ _rotation_from_hinge(axis, float(joints[index]))
    return position, rotation


def end_frames(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrist position in metres and wrist_yaw orientation, shapes ``(T, 2, 3)`` and ``(T, 2, 3, 3)``."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim == 1:
        joints = joints.reshape(1, 24)
    if joints.ndim != 2 or joints.shape[1] != 24 or joints.shape[0] == 0 or not np.isfinite(joints).all():
        raise ValueError("joints must be finite and non-empty (T, 24)")
    positions = np.empty((len(joints), 2, 3), dtype=np.float64)
    rotations = np.empty((len(joints), 2, 3, 3), dtype=np.float64)
    for row, sample in enumerate(joints):
        waist_pos, waist_rot = _body_pose(sample, WAIST_LINKS, np.eye(3), np.zeros(3))
        left_pos, left_rot = _body_pose(sample, LEFT_ARM_LINKS, waist_rot, waist_pos)
        right_pos, right_rot = _body_pose(sample, RIGHT_ARM_LINKS, waist_rot, waist_pos)
        positions[row, 0] = left_pos + left_rot @ LEFT_TCP_OFFSET
        positions[row, 1] = right_pos + right_rot @ RIGHT_TCP_OFFSET
        rotations[row, 0] = left_rot
        rotations[row, 1] = right_rot
    return positions, rotations
