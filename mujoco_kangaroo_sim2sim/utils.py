# Copyright (c) 2025, Fabio Amadio (fabioamadio93@gmail.com),
# Lorenzo Uttini (uttini.lorenzo@gmail.com).

# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import mujoco
import numpy as np
import torch

# ===========================
# === Rotation Utilities ====
# ===========================


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector `v` by the inverse of a quaternion `q`.

    Args:
        q: Quaternion in (w, x, y, z) format. Shape: (..., 4).
        v: Vector to rotate. Shape: (..., 3).

    Returns:
        Rotated vector. Shape: (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    if q_vec.dim() == 2:
        c = (
            q_vec
            * torch.bmm(
                q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)
            ).squeeze(-1)
            * 2.0
        )
    else:
        c = (
            q_vec
            * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1)
            * 2.0
        )
    return a - b + c


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector using a quaternion.

    Args:
        quat: Quaternion in (w, x, y, z) format. Shape: (..., 4).
        vec: Vector to rotate. Shape: (..., 3).

    Returns:
        Rotated vector. Shape: (..., 3).
    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    r"""Wrap input angles (radians) to the range [-π, π].

    Args:
        angles: Tensor of angles (in radians), any shape.

    Returns:
        Tensor of wrapped angles in the range [-π, π], same shape as input.
    """
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    return torch.where(
        (wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi
    )


# ===========================
# === MuJoCo Utilities ======
# ===========================


def dof_width(joint_type: mujoco.mjtJoint) -> int:
    """Return the dimensionality of a joint in `qvel` for a given joint type.

    Args:
        joint_type: MuJoCo joint type identifier.

    Returns:
        DOF width (int).
    """
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: mujoco.mjtJoint) -> int:
    """Return the dimensionality of a joint in `qpos` for a given joint type.

    Args:
        joint_type: MuJoCo joint type identifier.

    Returns:
        Position width (int).
    """
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def get_ids(
    mj_model: mujoco.MjModel, jnt_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get joint, qvel, and qpos indices for a list of joint names.

    Args:
        mj_model: MuJoCo model.
        jnt_names: List of joint names.

    Returns:
        Tuple of arrays: (joint_ids, qvel_ids, qpos_ids).
    """
    jnt_ids = []
    qvel_ids = []
    qpos_ids = []
    for joint_name in jnt_names:
        jnt_id = mj_model.joint(joint_name).id
        jnt_ids.append(jnt_id)
        jnt_type = mj_model.jnt_type[jnt_id]
        vadr = mj_model.jnt_dofadr[jnt_id]
        qadr = mj_model.jnt_qposadr[jnt_id]
        vdim = dof_width(jnt_type)
        qdim = qpos_width(jnt_type)
        qvel_ids.extend(range(vadr, vadr + vdim))
        qpos_ids.extend(range(qadr, qadr + qdim))
    return np.array(jnt_ids), np.array(qvel_ids), np.array(qpos_ids)


def set_tracking_viewer(
    mj_model: mujoco.MjModel, viewer: mujoco.viewer.Handle, body_name: str
) -> None:
    """Set MuJoCo viewer camera to track a specific body.

    Args:
        mj_model: MuJoCo model.
        viewer: Viewer handle.
        body_name: Name of the body to track.

    Returns:
        Never returns (modifies viewer in-place).
    """
    torso_body_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = torso_body_id
    viewer.cam.distance = 3
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20


# ===========================
# === Policy Utilities ======
# ===========================


def squash_action(
    action: torch.Tensor,
    tanh_offset: torch.Tensor,
    up_lim: torch.Tensor,
    low_lim: torch.Tensor,
) -> torch.Tensor:
    """Squash action using tanh and map it to motor command limits.

    Args:
        action: Raw action tensor.
        tanh_offset: Offset tensor added before tanh.
        up_lim: Upper limit for motor commands.
        low_lim: Lower limit for motor commands.

    Returns:
        Squashed action tensor within motor limits.
    """
    motor_cmd = torch.tanh(action * 0.1 + tanh_offset)
    return motor_cmd * (up_lim - low_lim) * 0.5 + (up_lim + low_lim) * 0.5


def foh(
    new_ref: float, old_ref: float, tot_steps: int, current_step: int
) -> float:
    """Perform First-Order Hold (FOH) interpolation.

    Args:
        new_ref: New reference value.
        old_ref: Old reference value.
        tot_steps: Total number of interpolation steps.
        current_step: Current step in the interpolation.

    Returns:
        Interpolated value at `current_step`.
    """
    if tot_steps <= 0:
        raise ValueError("Total steps must be greater than zero.")
    if current_step < 0 or current_step > tot_steps:
        raise ValueError("Current step must be between 0 and total steps.")

    slope = (new_ref - old_ref) / tot_steps
    return old_ref + slope * current_step


def compute_cmd_vel(
    x_vel_target: float,
    y_vel_target: float,
    root_link_quat_w: torch.Tensor,
    heading_target: float,
    ang_vel_range: tuple[float, float] = (-1.0, 1.0),
    heading_control_stiffness: float = 0.5,
) -> torch.Tensor:
    """Compute velocity command given desired heading and linear targets.

    Args:
        x_vel_target: Target velocity along x-axis.
        y_vel_target: Target velocity along y-axis.
        root_link_quat_w: Root orientation quaternion (w, x, y, z).
        heading_target: Desired heading angle (rad).
        ang_vel_range: Angular velocity range (min, max).
        heading_control_stiffness: Heading correction gain.

    Returns:
        Tensor of shape (3,) with [x_vel, y_vel, angular_vel].
    """
    dev = root_link_quat_w.device
    FORWARD_VEC_B = torch.tensor([1.0, 0.0, 0.0], device=dev)
    forward_w = quat_apply(root_link_quat_w, FORWARD_VEC_B)
    heading_w = torch.atan2(forward_w[1], forward_w[0])
    heading_error = wrap_to_pi(heading_target - heading_w)
    ang_vel = torch.clip(
        heading_control_stiffness * heading_error,
        min=ang_vel_range[0],
        max=ang_vel_range[1],
    )
    cmd_vel = torch.tensor([x_vel_target, y_vel_target, ang_vel], device=dev)
    return cmd_vel
