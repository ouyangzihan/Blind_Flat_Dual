# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Walk These Ways augmented auxiliary reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .gait_commands import GaitBehaviorCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Foot order must match GaitBehaviorCommand.foot_indices / desired_contact_states:
# [FL, FR, RL, RR]. The WTW source comment says "[FR, FL, RR, RL]" for Raibert, but
# the Y-sign pattern [+w/2, -w/2, +w/2, -w/2] is only correct for left-positive body Y
# when feet are ordered [FL, FR, RL, RR] (as in the actual Isaac Gym feet_indices used
# with that formula). Using the comment's FR-first order inverts lateral targets and
# encourages legs to cross through the body center.
FOOT_NAMES_CONTACT = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
FOOT_NAMES_RAIBERT = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def _gravity_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def _get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> GaitBehaviorCommand:
    return env.command_manager.get_term(command_name)


def wtw_tracking_contacts_shaped_force(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    gait_force_sigma: float,
) -> torch.Tensor:
    """Reward matching foot contact forces to the commanded gait schedule."""
    gait_cmd = _get_gait_command(env, command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    desired_contact = gait_cmd.desired_contact_states
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(4):
        reward += -(1.0 - desired_contact[:, i]) * (
            1.0 - torch.exp(-foot_forces[:, i] ** 2 / gait_force_sigma)
        )
    return (reward / 4.0) * _gravity_scale(env)


def wtw_tracking_contacts_shaped_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    gait_vel_sigma: float,
) -> torch.Tensor:
    """Penalize foot velocity during commanded stance phases."""
    gait_cmd = _get_gait_command(env, command_name)
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_velocities = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
    desired_contact = gait_cmd.desired_contact_states
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(4):
        reward += -desired_contact[:, i] * (1.0 - torch.exp(-foot_velocities[:, i] ** 2 / gait_vel_sigma))
    return (reward / 4.0) * _gravity_scale(env)


def wtw_feet_clearance_cmd_linear(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    foot_radius_offset: float,
) -> torch.Tensor:
    """Penalize deviation from commanded footswing height during swing phases."""
    gait_cmd = _get_gait_command(env, command_name)
    asset: RigidObject = env.scene[asset_cfg.name]
    phases = 1.0 - torch.abs(
        1.0 - torch.clip((gait_cmd.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0
    )
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    target_height = gait_cmd.command[:, 6].unsqueeze(1) * phases + foot_radius_offset
    rew_foot_clearance = torch.square(target_height - foot_height) * (1.0 - gait_cmd.desired_contact_states)
    return torch.sum(rew_foot_clearance, dim=1) * _gravity_scale(env)


def wtw_raibert_heuristic(
    env: ManagerBasedRLEnv,
    velocity_command_name: str,
    gait_command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize deviation from Raibert-style commanded foot placements."""
    gait_cmd = _get_gait_command(env, gait_command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_cmd = env.command_manager.get_command(velocity_command_name)

    cur_footsteps_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    root_quat_yaw_inv = math_utils.quat_conjugate(asset.data.root_quat_w)
    for i in range(4):
        # Isaac Lab quats are (w, x, y, z); use math_utils.quat_apply_yaw (not the
        # Isaac Gym xyzw pattern of zeroing quat[:, :2], which breaks yaw extraction).
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
            root_quat_yaw_inv,
            cur_footsteps_translated[:, i, :],
        )

    # Nominal placements for foot order [FL, FR, RL, RR] in body frame (Y left-positive).
    desired_stance_width = gait_cmd.command[:, 9:10]
    desired_ys_nom = torch.cat(
        [
            desired_stance_width / 2,
            -desired_stance_width / 2,
            desired_stance_width / 2,
            -desired_stance_width / 2,
        ],
        dim=1,
    )
    desired_stance_length = gait_cmd.command[:, 10:11]
    desired_xs_nom = torch.cat(
        [
            desired_stance_length / 2,
            desired_stance_length / 2,
            -desired_stance_length / 2,
            -desired_stance_length / 2,
        ],
        dim=1,
    )

    phases = torch.abs(1.0 - (gait_cmd.foot_indices * 2.0)) * 1.0 - 0.5
    frequencies = gait_cmd.command[:, 1]
    x_vel_des = velocity_cmd[:, 0:1]
    yaw_vel_des = velocity_cmd[:, 2:3]
    y_vel_des = yaw_vel_des * desired_stance_length / 2
    desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
    desired_ys_offset[:, 2:4] *= -1
    desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

    desired_ys_nom = desired_ys_nom + desired_ys_offset
    desired_xs_nom = desired_xs_nom + desired_xs_offset
    desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

    err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
    return torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2)) * _gravity_scale(env)


def wtw_orientation_control(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from commanded body pitch and roll."""
    gait_cmd = _get_gait_command(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch_commands = gait_cmd.command[:, 7:9]
    quat_roll = math_utils.quat_from_euler_xyz(
        torch.zeros_like(roll_pitch_commands[:, 1]),
        torch.zeros_like(roll_pitch_commands[:, 1]),
        -roll_pitch_commands[:, 1],
    )
    quat_pitch = math_utils.quat_from_euler_xyz(
        torch.zeros_like(roll_pitch_commands[:, 0]),
        -roll_pitch_commands[:, 0],
        torch.zeros_like(roll_pitch_commands[:, 0]),
    )
    desired_base_quat = math_utils.quat_mul(quat_roll, quat_pitch)
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    desired_projected_gravity = math_utils.quat_apply_inverse(desired_base_quat, gravity_vec)
    return (
        torch.sum(torch.square(asset.data.projected_gravity_b[:, :2] - desired_projected_gravity[:, :2]), dim=1)
        * _gravity_scale(env)
    )


def wtw_jump(
    env: ManagerBasedRLEnv,
    command_name: str,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of commanded body height offset."""
    gait_cmd = _get_gait_command(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = asset.data.root_pos_w[:, 2]
    jump_height_target = gait_cmd.command[:, 0] + base_height_target
    return -torch.square(body_height - jump_height_target) * _gravity_scale(env)
