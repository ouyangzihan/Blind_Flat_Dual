# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.wtw_rewards import FOOT_NAMES_CONTACT, FOOT_NAMES_RAIBERT
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg, CurriculumCfg

from .flat_env_cfg import UnitreeGo2WFlatEnvCfg
from .rough_env_cfg import UnitreeGo2WRewardsCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_WHEEL_CONTACT_TARGET_WEIGHT = 0.
_WHEEL_CONTACT_YAW_ONLY_TARGET_WEIGHT = -0. # -1.
_WHEEL_SPEED_YAW_ONLY_TARGET_WEIGHT = 0.
_WHEEL_LATERAL_DRAG_TARGET_WEIGHT = -1.5 
_UNCOMMANDED_BASE_VEL_PENALTY_TARGET_WEIGHT = -.0



def _yaw_only_command_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    lin_vel_threshold: float,
    yaw_vel_threshold: float,
) -> torch.Tensor:
    """Return a per-env float mask that is 1 when only the yaw command is non-zero.

    The base velocity command is ``[lin_vel_x, lin_vel_y, ang_vel_z]``; the mask is active
    when the planar linear command magnitude is below ``lin_vel_threshold`` while the yaw
    command magnitude is above ``yaw_vel_threshold`` (i.e. a commanded turn-in-place).
    """
    command = env.command_manager.get_command(command_name)
    lin_vel_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_vel_abs = torch.abs(command[:, 2])
    yaw_only = (lin_vel_norm < lin_vel_threshold) & (yaw_vel_abs > yaw_vel_threshold)
    return yaw_only.float()


def wheel_speed_yaw_only_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    lin_vel_threshold: float = 0.1,
    yaw_vel_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize wheel joint speed when only the yaw command is non-zero (turn-in-place)."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_speed = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return wheel_speed * _yaw_only_command_mask(env, command_name, lin_vel_threshold, yaw_vel_threshold)


def wheel_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize wheel-ground contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    num_in_contact = (forces > force_threshold).float().sum(dim=1)
    return num_in_contact


def wheel_contact_yaw_only_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    force_threshold: float = 1.0,
    lin_vel_threshold: float = 0.1,
    yaw_vel_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize wheel-ground contact when only the yaw command is non-zero (turn-in-place)."""
    return wheel_contact_penalty(env, sensor_cfg, force_threshold) * _yaw_only_command_mask(
        env, command_name, lin_vel_threshold, yaw_vel_threshold
    )


def wheel_lateral_drag_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize lateral wheel slip on the ground plane.

    Uses each wheel's world-frame linear velocity relative to the fixed ground, projected
    onto the horizontal plane, then expressed in the wheel link frame. Only the wheel-frame
    lateral (y) component is penalized; the rolling direction (wheel-frame x) is excluded.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > force_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    num_wheels = len(asset_cfg.body_ids)

    wheel_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].clone()
    wheel_vel_w[..., 2] = 0.0

    wheel_vel_local = torch.zeros(env.num_envs, num_wheels, 3, device=env.device)
    for i in range(num_wheels):
        wheel_vel_local[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.body_quat_w[:, asset_cfg.body_ids[i], :], wheel_vel_w[:, i, :]
        )

    lateral_drag = torch.square(wheel_vel_local[:, :, 1])
    penalty = torch.sum(lateral_drag * contacts.float(), dim=1)
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return penalty


def uncommanded_base_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    ang_vel_scale: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base velocity on axes whose velocity command is near zero.

    The base velocity command is ``[lin_vel_x, lin_vel_y, ang_vel_z]``. Each axis is
    penalized only when the corresponding command magnitude is below ``command_threshold``.
    Yaw angular velocity is scaled by ``ang_vel_scale`` relative to the linear axes.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel = asset.data.root_lin_vel_b
    ang_vel = asset.data.root_ang_vel_b

    mask_x = (torch.abs(command[:, 0]) < command_threshold).float()
    mask_y = (torch.abs(command[:, 1]) < command_threshold).float()
    mask_yaw = (torch.abs(command[:, 2]) < command_threshold).float()

    penalty = (
        mask_x * torch.square(lin_vel[:, 0])
        + mask_y * torch.square(lin_vel[:, 1])
        + ang_vel_scale * mask_yaw * torch.square(ang_vel[:, 2])
    )
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return penalty


_VEL_TRACKING_LIN_XY_TARGET_WEIGHT = 3.0
_VEL_TRACKING_ANG_Z_TARGET_WEIGHT = 1.5
_UPWARD_TARGET_WEIGHT = 1.0
_ACTION_RATE_L2_TARGET_WEIGHT = -0.01 # -0.0075
_JOINT_POS_PENALTY_TARGET_WEIGHT = -0.25
_WTW_TRACKING_FORCE_TARGET_WEIGHT = 1.0000000000 # 0.627027027 # .4
_WTW_TRACKING_VEL_TARGET_WEIGHT = 2.9052128439 # 1.067280046 # .4
_WTW_FEET_CLEARANCE_TARGET_WEIGHT = -63.4512787287 # -13.36127328 # -3.0
_WTW_RAIBERT_TARGET_WEIGHT = -5.7565416815
_WTW_ORIENTATION_TARGET_WEIGHT = -7.4647454989 # -1.899396692 # -.5
_WTW_JUMP_TARGET_WEIGHT = 30.2176480985 # 5.509803415 # 1.0
_WTW_REWARD_TERM_NAMES = (
    "wtw_tracking_contacts_shaped_force",
    "wtw_tracking_contacts_shaped_vel",
    "wtw_feet_clearance_cmd_linear",
    "wtw_raibert_heuristic",
    "wtw_orientation_control",
    "wtw_jump",
)
_VEL_TRACKING_START_SCALE = 2.
_VEL_TRACKING_WEIGHT_DECAY_ITERATIONS = 500
_VEL_TRACKING_STEPS_PER_ITERATION = 24  # matches UnitreeGo2WFlatLegsPPORunnerCfg.num_steps_per_env


@configclass
class UnitreeGo2WFlatLegsCommandsCfg(CommandsCfg):
    """Commands that require at least one non-zero lateral or yaw velocity component."""

    base_velocity = mdp.UniformLateralYawVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        lateral_yaw_threshold=0.1,
        max_resample_attempts=100,
        ranges=mdp.UniformLateralYawVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
    gait_behavior = mdp.GaitBehaviorCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        kappa_gait_probs=0.07,
        ranges=mdp.GaitBehaviorCommandCfg.Ranges(
            body_height=(0., 0.05),
            frequency=(2.0, 3.0),
            phase=(0.5, 0.5),
            offset=(0.0, 0.0),
            bound=(0.0, 0.0),
            duration=(0.5, 0.5),
            footswing_height=(0.03, 0.05),
            body_pitch=(0,0), # (-0.1, 0.1),
            body_roll=(0,0), # (-0.1, 0.1),
            stance_width=(.25, .31), # (0.23, 0.33),
            stance_length=(.36, .42), # (0.34, 0.44),
        ),
    )


@configclass
class UnitreeGo2WFlatLegsRewardsCfg(UnitreeGo2WRewardsCfg):
    """Walk These Ways augmented auxiliary rewards for leg-raising locomotion."""

    wtw_tracking_contacts_shaped_force = RewTerm(
        func=mdp.wtw_tracking_contacts_shaped_force,
        weight=0.0,
        params={
            "command_name": "gait_behavior",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_NAMES_CONTACT),
            "gait_force_sigma": 100.0,
        },
    )
    wtw_tracking_contacts_shaped_vel = RewTerm(
        func=mdp.wtw_tracking_contacts_shaped_vel,
        weight=0.0,
        params={
            "command_name": "gait_behavior",
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES_CONTACT),
            "gait_vel_sigma": 10.0,
        },
    )
    wtw_feet_clearance_cmd_linear = RewTerm(
        func=mdp.wtw_feet_clearance_cmd_linear,
        weight=0.0,
        params={
            "command_name": "gait_behavior",
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES_CONTACT),
            "foot_radius_offset": 0.086,
        },
    )
    wtw_raibert_heuristic = RewTerm(
        func=mdp.wtw_raibert_heuristic,
        weight=0.0,
        params={
            "velocity_command_name": "base_velocity",
            "gait_command_name": "gait_behavior",
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES_RAIBERT, preserve_order=True),
        },
    )
    wtw_orientation_control = RewTerm(
        func=mdp.wtw_orientation_control,
        weight=0.0,
        params={"command_name": "gait_behavior"},
    )
    wtw_jump = RewTerm(
        func=mdp.wtw_jump,
        weight=0.0,
        params={
            "command_name": "gait_behavior",
            "base_height_target": 0.40,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    wheel_speed_yaw_only = RewTerm(
        func=wheel_speed_yaw_only_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "lin_vel_threshold": 0.1,
            "yaw_vel_threshold": 0.1,
        },
    )
    wheel_contact = RewTerm(
        func=wheel_contact_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "force_threshold": 1.0,
        },
    )
    wheel_contact_yaw_only = RewTerm(
        func=wheel_contact_yaw_only_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "force_threshold": 1.0,
            "lin_vel_threshold": 0.1,
            "yaw_vel_threshold": 0.1,
        },
    )
    wheel_lateral_drag = RewTerm(
        func=wheel_lateral_drag_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "force_threshold": 1.0,
        },
    )
    uncommanded_base_vel = RewTerm(
        func=uncommanded_base_vel_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "ang_vel_scale": 5.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    lin_acc_z_l2 = RewTerm(
        func=mdp.lin_acc_z_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )
    ang_acc_xy_l2 = RewTerm(
        func=mdp.ang_acc_xy_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )


@configclass
class UnitreeGo2WFlatLegsCurriculumCfg(CurriculumCfg):
    """Curriculum for flat-legs training; inherits disabled terrain/command curricula from flat env."""

    track_lin_vel_xy_weight = CurrTerm(
        func=mdp.reward_weight_linear_decay,
        params={
            "term_name": "track_lin_vel_xy_exp",
            "target_weight": _VEL_TRACKING_LIN_XY_TARGET_WEIGHT,
            "start_scale": _VEL_TRACKING_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    track_ang_vel_z_weight = CurrTerm(
        func=mdp.reward_weight_linear_decay,
        params={
            "term_name": "track_ang_vel_z_exp",
            "target_weight": _VEL_TRACKING_ANG_Z_TARGET_WEIGHT,
            "start_scale": _VEL_TRACKING_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    upward_weight = CurrTerm(
        func=mdp.reward_weight_linear_decay,
        params={
            "term_name": "upward",
            "target_weight": _UPWARD_TARGET_WEIGHT,
            "start_scale": _VEL_TRACKING_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    action_rate_l2_weight = CurrTerm(
        func=mdp.reward_weight_linear_decay,
        params={
            "term_name": "action_rate_l2",
            "target_weight": _ACTION_RATE_L2_TARGET_WEIGHT,
            "start_scale": _VEL_TRACKING_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    joint_pos_penalty_weight = CurrTerm(
        func=mdp.reward_weight_linear_decay,
        params={
            "term_name": "joint_pos_penalty",
            "target_weight": _JOINT_POS_PENALTY_TARGET_WEIGHT,
            "start_scale": _VEL_TRACKING_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_tracking_force_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_tracking_contacts_shaped_force",
            "target_weight": _WTW_TRACKING_FORCE_TARGET_WEIGHT,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_tracking_vel_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_tracking_contacts_shaped_vel",
            "target_weight": _WTW_TRACKING_VEL_TARGET_WEIGHT,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_feet_clearance_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_feet_clearance_cmd_linear",
            "target_weight": _WTW_FEET_CLEARANCE_TARGET_WEIGHT,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_raibert_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_raibert_heuristic",
            "target_weight": _WTW_RAIBERT_TARGET_WEIGHT,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_orientation_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_orientation_control",
            "target_weight": _WTW_ORIENTATION_TARGET_WEIGHT,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_jump_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_jump",
            "target_weight": _WTW_JUMP_TARGET_WEIGHT,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )


@configclass
class UnitreeGo2WFlatLegsEnvCfg(UnitreeGo2WFlatEnvCfg):
    """Flat terrain env for Go2W with mixed commands that require leg raising."""

    commands: UnitreeGo2WFlatLegsCommandsCfg = UnitreeGo2WFlatLegsCommandsCfg()
    rewards: UnitreeGo2WFlatLegsRewardsCfg = UnitreeGo2WFlatLegsRewardsCfg()
    curriculum: UnitreeGo2WFlatLegsCurriculumCfg = UnitreeGo2WFlatLegsCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Rewards------------------------------
        self.rewards.action_rate_l2.weight = _ACTION_RATE_L2_TARGET_WEIGHT * _VEL_TRACKING_START_SCALE

        # Weaken default-pose penalties to allow leg raising.
        # self.rewards.stand_still.weight = -0.5
        self.rewards.joint_pos_penalty.weight = _JOINT_POS_PENALTY_TARGET_WEIGHT * _VEL_TRACKING_START_SCALE

        # Stable base (orientation handled by WTW orientation_control).
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -4.0  # 2x inherited weight (-2.0)
        self.rewards.ang_vel_xy_l2.weight = -0.1  # 2x inherited weight (-0.05)
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.lin_acc_z_l2.weight = -.004 
        self.rewards.lin_acc_z_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.ang_acc_xy_l2.weight = -0.00005 
        self.rewards.ang_acc_xy_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_acc_l2.weight = -5.0e-7 
        self.rewards.joint_acc_wheel_l2.weight = -5.0e-9 
        # self.rewards.contact_forces.weight = -5.0e-4 # -1.5e-4

        # Diagonal symmetry.
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.action_mirror.weight = -0.05
        self.rewards.action_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Disable legacy gait/stance rewards superseded by Walk These Ways terms.
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_distance_y_exp.weight = 0.0
        self.rewards.feet_distance_xy_exp.weight = 0.0

        # Selected rewards start scaled up and linearly decay to 100% by iteration 500 (see curriculum).
        self.rewards.track_lin_vel_xy_exp.weight = _VEL_TRACKING_LIN_XY_TARGET_WEIGHT * _VEL_TRACKING_START_SCALE
        self.rewards.track_ang_vel_z_exp.weight = _VEL_TRACKING_ANG_Z_TARGET_WEIGHT * _VEL_TRACKING_START_SCALE
        self.rewards.upward.weight = _UPWARD_TARGET_WEIGHT * _VEL_TRACKING_START_SCALE

        # Walk These Ways augmented auxiliary rewards ramp from 0% to 100% by iteration 500 (see curriculum).
        for term_name in _WTW_REWARD_TERM_NAMES:
            getattr(self.rewards, term_name).weight = 0.0

        # Penalize wheel usage when only a yaw command is active (turn-in-place should use legs).
        self.rewards.wheel_speed_yaw_only.weight = _WHEEL_SPEED_YAW_ONLY_TARGET_WEIGHT
        self.rewards.wheel_speed_yaw_only.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_contact.weight = _WHEEL_CONTACT_TARGET_WEIGHT
        self.rewards.wheel_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_contact_yaw_only.weight = _WHEEL_CONTACT_YAW_ONLY_TARGET_WEIGHT
        self.rewards.wheel_contact_yaw_only.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_lateral_drag.weight = _WHEEL_LATERAL_DRAG_TARGET_WEIGHT
        self.rewards.wheel_lateral_drag.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_lateral_drag.params["asset_cfg"].body_names = [self.foot_link_name]

        # Penalize base motion on axes with zero velocity command (yaw weighted 5x heavier).
        self.rewards.uncommanded_base_vel.weight = _UNCOMMANDED_BASE_VEL_PENALTY_TARGET_WEIGHT

        if self.__class__.__name__ == "UnitreeGo2WFlatLegsEnvCfg":
            self.disable_zero_weight_rewards()

    def disable_zero_weight_rewards(self):
        """Disable zero-weight rewards, but keep curriculum-ramped WTW terms registered."""
        for attr in dir(self.rewards):
            if not attr.startswith("__") and attr not in _WTW_REWARD_TERM_NAMES:
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr is not None and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)
