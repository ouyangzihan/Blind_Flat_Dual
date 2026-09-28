# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.wtw_rewards import FOOT_NAMES_CONTACT, FOOT_NAMES_RAIBERT
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg, CurriculumCfg, ObservationsCfg

from .flat_env_cfg import UnitreeGo2WFlatEnvCfg
from .rough_env_cfg import UnitreeGo2WRewardsCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_WHEEL_CONTACT_TARGET_WEIGHT = 0.
_WHEEL_CONTACT_YAW_ONLY_TARGET_WEIGHT = -0. # -1.
_WHEEL_SPEED_YAW_ONLY_TARGET_WEIGHT = 0.
_WHEEL_LATERAL_DRAG_TARGET_WEIGHT = -0.  
_FEET_LATERAL_REGULATION_TARGET_WEIGHT = -0.0 # -0.1
_FEET_LATERAL_REGULATION_CLEARANCE_SCALE = 200.0
_FEET_LATERAL_REGULATION_FOOT_RADIUS = 0.086
_UNCOMMANDED_BASE_VEL_PENALTY_TARGET_WEIGHT = -.0
_UNCOMMANDED_ANG_VEL_Z_TARGET_WEIGHT = -1 # -10.0
# Near-zero gate for base_velocity axes (vx/vy/yaw) and gait plant mode.
_CMD_NEAR_ZERO_THRESHOLD = 0.05
# Continuous left–right calf clearance (normalized squared); softer than binary contact at -500.
_CALF_CLEARANCE_TARGET_WEIGHT = -20000.0
# Soft separation at the calflower sample point (m). URDF calflower boxes are ~2.2 cm wide, so
# mesh contact is ~0.022 m between centers; 0.03 only penalizes near-contact, not wide swings.
_CALF_CLEARANCE_MIN_DIST = 0.05
# Ordered [FR, FL, RR, RL]; only pairs (0,1) and (2,3) are penalized (front / rear).
_CALF_CLEARANCE_BODY_NAMES = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
# URDF ``*_calflower_joint`` origin in the calf frame (calflower bodies are merged into calf).
_CALF_CLEARANCE_OFFSET_B = (0.02, 0.0, -0.148)



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
    lin_vel_threshold: float = _CMD_NEAR_ZERO_THRESHOLD,
    yaw_vel_threshold: float = _CMD_NEAR_ZERO_THRESHOLD,
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
    lin_vel_threshold: float = _CMD_NEAR_ZERO_THRESHOLD,
    yaw_vel_threshold: float = _CMD_NEAR_ZERO_THRESHOLD,
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


def feet_lateral_regulation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_radius: float = _FEET_LATERAL_REGULATION_FOOT_RADIUS,
    clearance_scale: float = _FEET_LATERAL_REGULATION_CLEARANCE_SCALE,
) -> torch.Tensor:
    """Penalize lateral foot/wheel motion near the ground via a soft clearance gate.

    Same wheel-frame lateral velocity as ``wheel_lateral_drag_penalty`` (absolute, not
    squared), gated by ``exp(-clearance_scale * x)`` where ``x = max(0, body_z - foot_radius)``
    is wheel clearance to flat ground. Full penalty at contact (x=0); decays as the foot lifts.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_feet = len(asset_cfg.body_ids)

    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].clone()
    foot_vel_w[..., 2] = 0.0

    foot_vel_local = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    for i in range(num_feet):
        foot_vel_local[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.body_quat_w[:, asset_cfg.body_ids[i], :], foot_vel_w[:, i, :]
        )

    lateral_speed = torch.abs(foot_vel_local[:, :, 1])
    clearance = torch.clamp(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, min=0.0)
    proximity = torch.exp(-clearance_scale * clearance)
    penalty = torch.sum(lateral_speed * proximity, dim=1)
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return penalty


def calf_clearance_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float = _CALF_CLEARANCE_MIN_DIST,
    offset_b: tuple[float, float, float] = _CALF_CLEARANCE_OFFSET_B,
) -> torch.Tensor:
    """Penalize left–right calf pairs when calflower sample points are closer than ``min_dist``.

    Expects ``asset_cfg.body_names`` ordered as ``[FR_calf, FL_calf, RR_calf, RL_calf]``.
    Only the front (0–1) and rear (2–3) pairs are checked; front–rear pairs are ignored.

    Go2W spawns with ``merge_fixed_joints=True``, so ``*_calflower`` links are not separate
    bodies. Distance is measured between points offset by ``offset_b`` in each calf frame
    (URDF ``*_calflower_joint`` origin), where lower-calf collisions actually occur. Proximal
    calf joint origins stay ~hip-width apart even when those meshes intersect.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_bodies = len(body_ids)
    offset = torch.tensor(offset_b, device=env.device, dtype=torch.float32)
    pos = torch.zeros(env.num_envs, num_bodies, 3, device=env.device)
    for i in range(num_bodies):
        pos[:, i, :] = asset.data.body_pos_w[:, body_ids[i], :] + math_utils.quat_apply(
            asset.data.body_quat_w[:, body_ids[i], :],
            offset.expand(env.num_envs, 3),
        )
    penalty = torch.zeros(env.num_envs, device=env.device)
    for i, j in ((0, 1), (2, 3)):
        dist = torch.norm(pos[:, i] - pos[:, j], dim=-1)
        violation = torch.clamp(min_dist - dist, min=0.0) / min_dist
        penalty += torch.square(violation)
    penalty *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return penalty


def uncommanded_base_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = _CMD_NEAR_ZERO_THRESHOLD,
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


def uncommanded_ang_vel_z_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    yaw_vel_threshold: float = _CMD_NEAR_ZERO_THRESHOLD,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base yaw angular velocity when the yaw command is near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    mask_yaw = (torch.abs(command[:, 2]) < yaw_vel_threshold).float()
    penalty = mask_yaw * torch.square(asset.data.root_ang_vel_b[:, 2])
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return penalty


_VEL_TRACKING_LIN_XY_TARGET_WEIGHT = 3.0
_VEL_TRACKING_ANG_Z_TARGET_WEIGHT = 1.5
_VEL_TRACKING_TRANSITION_STD_SCALE = 2.0
_VEL_TRACKING_TRANSITION_DURATION_S = 0.75
_UPWARD_TARGET_WEIGHT = 1.
_ACTION_RATE_L2_TARGET_WEIGHT = -0.005 #-0.01
# Start softer (-0.0075) and ramp magnitude to target by iteration 500.
_ACTION_RATE_L2_START_SCALE = 0.5
_ACTION_SMOOTHNESS_L2_TARGET_WEIGHT = -0.0005 #-0.001
_ANG_VEL_XY_L2_TARGET_WEIGHT = -0.4 # -0.2
# Start softer (-0.1) and ramp magnitude to target by iteration 500.
_ANG_VEL_XY_L2_START_SCALE = 0.5
_JOINT_POS_PENALTY_TARGET_WEIGHT = -0.25 # -0.5
_WTW_TRACKING_FORCE_TARGET_WEIGHT = 1.5 # 1.0000000000 # 0.627027027 # .4
_WTW_TRACKING_VEL_TARGET_WEIGHT = 2.9052128439 # 1.067280046 # .4
_WTW_FEET_CLEARANCE_TARGET_WEIGHT = -30 # -63.4512787287 # -13.36127328 # -3.0
_WTW_RAIBERT_TARGET_WEIGHT = -10 # -20 # -1.899396692 # -.5
_WTW_ORIENTATION_TARGET_WEIGHT = -7.4647454989 # -1.899396692 # -.5
_WTW_JUMP_TARGET_WEIGHT = 30.2176480985 # 5.509803415 # 1.0
# WTW terms start at this fraction of target and ramp to 100% by iteration 1000.
_WTW_START_SCALE = 0.5
_WTW_REWARD_TERM_NAMES = (
    "wtw_tracking_contacts_shaped_force",
    "wtw_tracking_contacts_shaped_vel",
    "wtw_feet_clearance_cmd_linear",
    "wtw_raibert_heuristic",
    "wtw_orientation_control",
    "wtw_jump",
)
# Reward terms that curriculum-ramp; must stay registered even when initial weight is 0.
_CURRICULUM_RAMPED_REWARD_TERM_NAMES = _WTW_REWARD_TERM_NAMES + ("uncommanded_ang_vel_z", "calf_clearance")
_VEL_TRACKING_START_SCALE = 1.5
_VEL_TRACKING_WEIGHT_DECAY_ITERATIONS = 1000
_VEL_TRACKING_STEPS_PER_ITERATION = 24  # matches UnitreeGo2WFlatLegsPPORunnerCfg.num_steps_per_env
# Footswing command starts at this fraction of target and ramps to 100% over 2x the WTW weight interval.
_FOOTSWING_HEIGHT_TARGET = (0.02, 0.06)
_FOOTSWING_HEIGHT_START_SCALE = 1
_FOOTSWING_HEIGHT_RAMP_ITERATIONS = 4 * _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS
# PPO knobs (flat-legs only; wired in UnitreeGo2WFlatLegsPPORunnerCfg).
_PPO_LEARNING_RATE = 1.0e-3
# 0.005 for learning iterations [0, 10000), then 0.001. (Previously a flat 0.01.)
_PPO_ENTROPY_COEF = 0.005  # 0.01
_PPO_ENTROPY_COEF_LATE = 0.001
_PPO_ENTROPY_DROP_ITERATION = 20000


@configclass
class UnitreeGo2WFlatLegsCommandsCfg(CommandsCfg):
    """Velocity and gait-behavior commands for flat-legs training."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.75, 0.75), ang_vel_z=(-1.5, 1.5), heading=(-math.pi, math.pi)
        ),
    )
    gait_behavior = mdp.GaitBehaviorCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        kappa_gait_probs=0.07,
        # freq = frequency_base * (1 + |vy| * frequency_vy_scale + |yaw| * frequency_yaw_scale).
        # Planted gait (footswing=0, all-stance) is wired from enable_wtw_planted_gait.
        velocity_command_name="base_velocity",
        frequency_base=(1, 1.5),
        frequency_vy_scale=0.5,
        frequency_yaw_scale=0.25,
        cmd_threshold=_CMD_NEAR_ZERO_THRESHOLD,
        ranges=mdp.GaitBehaviorCommandCfg.Ranges(
            body_height=(0.0, 0.0),
            frequency=(1.0, 1.5),
            phase=(0.5, 0.5),
            offset=(0.0, 0.0),
            bound=(0.0, 0.0),
            duration=(0.5, 0.5),
            footswing_height=_FOOTSWING_HEIGHT_TARGET,
            body_pitch=(0,0), # (-0.1, 0.1),
            body_roll=(0,0), # (-0.1, 0.1),
            stance_width=(.284, .284), # (0.23, 0.33),
            stance_length=(.387, .387), # (0.34, 0.44),
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
            "velocity_command_name": "base_velocity",
            "cmd_threshold": _CMD_NEAR_ZERO_THRESHOLD,
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
            # Overwritten from UnitreeGo2WFlatLegsEnvCfg.wtw_raibert_include_lin_vel_y.
            "include_lin_vel_y": False,
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
            "base_height_target": 0.408,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    wheel_speed_yaw_only = RewTerm(
        func=wheel_speed_yaw_only_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "lin_vel_threshold": _CMD_NEAR_ZERO_THRESHOLD,
            "yaw_vel_threshold": _CMD_NEAR_ZERO_THRESHOLD,
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
            "lin_vel_threshold": _CMD_NEAR_ZERO_THRESHOLD,
            "yaw_vel_threshold": _CMD_NEAR_ZERO_THRESHOLD,
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
    feet_lateral_regulation = RewTerm(
        func=feet_lateral_regulation_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "foot_radius": _FEET_LATERAL_REGULATION_FOOT_RADIUS,
            "clearance_scale": _FEET_LATERAL_REGULATION_CLEARANCE_SCALE,
        },
    )
    uncommanded_base_vel = RewTerm(
        func=uncommanded_base_vel_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": _CMD_NEAR_ZERO_THRESHOLD,
            "ang_vel_scale": 5.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    uncommanded_ang_vel_z = RewTerm(
        func=uncommanded_ang_vel_z_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "yaw_vel_threshold": _CMD_NEAR_ZERO_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    calf_clearance = RewTerm(
        func=calf_clearance_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=list(_CALF_CLEARANCE_BODY_NAMES),
                preserve_order=True,
            ),
            "min_dist": _CALF_CLEARANCE_MIN_DIST,
            "offset_b": _CALF_CLEARANCE_OFFSET_B,
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
    body_height_default = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": None,
            "target_height": .4,
        },
    )
    hip_joint_default = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.05,
            "full_stand_still_scale": 10.0,
        },
    )
    action_smoothness_l2 = RewTerm(func=mdp.ActionSmoothnessL2, weight=0.0)


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
            "start_scale": _ACTION_RATE_L2_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    action_smoothness_l2_weight = CurrTerm(
        func=mdp.reward_weight_linear_decay,
        params={
            "term_name": "action_smoothness_l2",
            "target_weight": _ACTION_SMOOTHNESS_L2_TARGET_WEIGHT,
            "start_scale": _ACTION_RATE_L2_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    ang_vel_xy_l2_weight = CurrTerm(
        func=mdp.reward_weight_linear_decay,
        params={
            "term_name": "ang_vel_xy_l2",
            "target_weight": _ANG_VEL_XY_L2_TARGET_WEIGHT,
            "start_scale": _ANG_VEL_XY_L2_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_tracking_force_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_tracking_contacts_shaped_force",
            "target_weight": _WTW_TRACKING_FORCE_TARGET_WEIGHT,
            "start_scale": _WTW_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_tracking_vel_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_tracking_contacts_shaped_vel",
            "target_weight": _WTW_TRACKING_VEL_TARGET_WEIGHT,
            "start_scale": _WTW_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_feet_clearance_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_feet_clearance_cmd_linear",
            "target_weight": _WTW_FEET_CLEARANCE_TARGET_WEIGHT,
            "start_scale": _WTW_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_raibert_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_raibert_heuristic",
            "target_weight": _WTW_RAIBERT_TARGET_WEIGHT,
            "start_scale": _WTW_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_orientation_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_orientation_control",
            "target_weight": _WTW_ORIENTATION_TARGET_WEIGHT,
            "start_scale": _WTW_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    wtw_jump_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "wtw_jump",
            "target_weight": _WTW_JUMP_TARGET_WEIGHT,
            "start_scale": _WTW_START_SCALE,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    uncommanded_ang_vel_z_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "uncommanded_ang_vel_z",
            "target_weight": _UNCOMMANDED_ANG_VEL_Z_TARGET_WEIGHT,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    calf_clearance_weight = CurrTerm(
        func=mdp.reward_weight_linear_ramp,
        params={
            "term_name": "calf_clearance",
            "target_weight": _CALF_CLEARANCE_TARGET_WEIGHT,
            "num_iterations": _VEL_TRACKING_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )
    footswing_height = CurrTerm(
        func=mdp.footswing_height_linear_ramp,
        params={
            "command_name": "gait_behavior",
            "target_range": _FOOTSWING_HEIGHT_TARGET,
            "start_scale": _FOOTSWING_HEIGHT_START_SCALE,
            "num_iterations": _FOOTSWING_HEIGHT_RAMP_ITERATIONS,
            "steps_per_iteration": _VEL_TRACKING_STEPS_PER_ITERATION,
        },
    )


@configclass
class UnitreeGo2WFlatLegsPolicyObsCfg(ObservationsCfg.PolicyCfg):
    """Policy observations with WTW gait clock and footswing appended."""

    gait_clock = ObsTerm(
        func=mdp.gait_clock,
        params={"command_name": "gait_behavior"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    gait_footswing = ObsTerm(
        func=mdp.gait_footswing,
        params={"command_name": "gait_behavior"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class UnitreeGo2WFlatLegsCriticObsCfg(ObservationsCfg.CriticCfg):
    """Critic observations with WTW gait clock and footswing appended."""

    gait_clock = ObsTerm(
        func=mdp.gait_clock,
        params={"command_name": "gait_behavior"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    gait_footswing = ObsTerm(
        func=mdp.gait_footswing,
        params={"command_name": "gait_behavior"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class UnitreeGo2WFlatLegsObservationsCfg(ObservationsCfg):
    """Flat-legs observations; gait terms can be detached via ``enable_wtw_gait_obs``."""

    policy: UnitreeGo2WFlatLegsPolicyObsCfg = UnitreeGo2WFlatLegsPolicyObsCfg()
    critic: UnitreeGo2WFlatLegsCriticObsCfg = UnitreeGo2WFlatLegsCriticObsCfg()


@configclass
class UnitreeGo2WFlatLegsEnvCfg(UnitreeGo2WFlatEnvCfg):
    """Flat terrain env for Go2W with mixed commands that require leg raising."""

    commands: UnitreeGo2WFlatLegsCommandsCfg = UnitreeGo2WFlatLegsCommandsCfg()
    observations: UnitreeGo2WFlatLegsObservationsCfg = UnitreeGo2WFlatLegsObservationsCfg()
    rewards: UnitreeGo2WFlatLegsRewardsCfg = UnitreeGo2WFlatLegsRewardsCfg()
    curriculum: UnitreeGo2WFlatLegsCurriculumCfg = UnitreeGo2WFlatLegsCurriculumCfg()
    # When False, reward weights stay at their target values (no iteration-based schedule).
    enable_reward_weight_curriculum: bool = True
    # When True, Raibert Y targets include commanded vy (full form). Default False matches WTW
    # (yaw-only Y offset) and avoids fighting height / lateral regulation while learning vy.
    wtw_raibert_include_lin_vel_y: bool = False
    # When True, |vy| and |yaw| below cmd_threshold force footswing=0 and all-stance
    # (pure-vx / standing plant). Set False to keep the same WTW gait for every command.
    enable_wtw_planted_gait: bool = True
    # When False, drop gait_clock and gait_footswing from policy/critic (57-dim obs, old checkpoints).
    enable_wtw_gait_obs: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Clearance uses calf link positions; PhysX self-collisions / contact sensors not needed.
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        self.rewards.wtw_raibert_heuristic.params["include_lin_vel_y"] = self.wtw_raibert_include_lin_vel_y
        self.commands.gait_behavior.enable_planted_gait = self.enable_wtw_planted_gait
        if not self.enable_wtw_gait_obs:
            self.observations.policy.gait_clock = None
            self.observations.policy.gait_footswing = None
            self.observations.critic.gait_clock = None
            self.observations.critic.gait_footswing = None

        # Log WTW episode means stratified by frequency_base bin (training signal unchanged).
        self.events.log_wtw_frequency_bin_rewards = EventTerm(
            func=mdp.enable_wtw_frequency_bin_reward_logging,
            mode="startup",
        )

        # ------------------------------Rewards------------------------------
        # Decay terms start scaled up and WTW terms ramp from 50% unless curriculum is disabled
        # (then all stay at their target weights).
        reward_weight_scale = _VEL_TRACKING_START_SCALE if self.enable_reward_weight_curriculum else 1.0
        action_rate_weight_scale = _ACTION_RATE_L2_START_SCALE if self.enable_reward_weight_curriculum else 1.0
        ang_vel_xy_weight_scale = _ANG_VEL_XY_L2_START_SCALE if self.enable_reward_weight_curriculum else 1.0
        wtw_weight_scale = _WTW_START_SCALE if self.enable_reward_weight_curriculum else 1.0
        # Non-WTW ramped terms still start at 0.
        ramp_from_zero_scale = 0.0 if self.enable_reward_weight_curriculum else 1.0
        # Footswing command starts at 20% of target and ramps to 100% by iteration 2000.
        footswing_scale = _FOOTSWING_HEIGHT_START_SCALE if self.enable_reward_weight_curriculum else 1.0
        lo, hi = _FOOTSWING_HEIGHT_TARGET
        self.commands.gait_behavior.ranges.footswing_height = (lo * footswing_scale, hi * footswing_scale)

        # Action-rate starts at 50% of target (-0.0075) and ramps to -0.01 by iteration 1000.
        self.rewards.action_rate_l2.weight = _ACTION_RATE_L2_TARGET_WEIGHT * action_rate_weight_scale
        # Action-smoothness starts at 50% of target (-0.0025) and ramps to -0.005 by iteration 1000.
        self.rewards.action_smoothness_l2.weight = _ACTION_SMOOTHNESS_L2_TARGET_WEIGHT * action_rate_weight_scale

        # Weaken default-pose penalties to allow leg raising.
        # self.rewards.stand_still.weight = -0.5
        self.rewards.joint_pos_penalty.weight = _JOINT_POS_PENALTY_TARGET_WEIGHT * 0.25
        self.rewards.hip_joint_default.weight = _JOINT_POS_PENALTY_TARGET_WEIGHT
        self.rewards.hip_joint_default.params["asset_cfg"].joint_names = [".*_hip_joint"]

        # Stable base (orientation handled by WTW orientation_control).
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -2.0 # -1.0 # -2.0  # 2x inherited weight (-2.0)
        # Starts at 50% of target (-0.1) and ramps to -0.2 by iteration 500.
        self.rewards.ang_vel_xy_l2.weight = _ANG_VEL_XY_L2_TARGET_WEIGHT * ang_vel_xy_weight_scale
        self.rewards.body_lin_acc_l2.weight = -0.005 # -0.01
        self.rewards.lin_acc_z_l2.weight = -0.004 # -.002 # -.004 
        self.rewards.lin_acc_z_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.ang_acc_xy_l2.weight = -0.0001 # -0.00005 
        self.rewards.ang_acc_xy_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_height_default.weight = -0.
        self.rewards.body_height_default.params["target_height"] = 0.4
        self.rewards.body_height_default.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_acc_l2.weight = -1.0e-6 #-5.0e-7
        self.rewards.joint_acc_wheel_l2.weight = -5.0e-8 
        self.rewards.joint_torques_l2.weight = -2.5e-5
        # self.rewards.undesired_contacts.weight = -2.5  # inherited weight (-1.0)
        self.rewards.contact_forces.weight = -0 # -1.5e-4
        # Calf clearance ramps from 0 → -20000 by iteration 1000.
        self.rewards.calf_clearance.weight = _CALF_CLEARANCE_TARGET_WEIGHT * ramp_from_zero_scale

        # Diagonal symmetry.
        self.rewards.joint_mirror.weight = -0.0 # -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.action_mirror.weight = -0.0 # -0.02
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

        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0

        # Selected rewards start scaled up and linearly decay to 100% by iteration 500 (see curriculum).
        self.rewards.track_lin_vel_xy_exp.weight = _VEL_TRACKING_LIN_XY_TARGET_WEIGHT * reward_weight_scale
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.5)
        self.rewards.track_lin_vel_xy_exp.params["transition_std_scale"] = _VEL_TRACKING_TRANSITION_STD_SCALE
        self.rewards.track_lin_vel_xy_exp.params["transition_duration_s"] = _VEL_TRACKING_TRANSITION_DURATION_S
        self.rewards.track_ang_vel_z_exp.weight = _VEL_TRACKING_ANG_Z_TARGET_WEIGHT * reward_weight_scale
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.5)
        self.rewards.track_ang_vel_z_exp.params["transition_std_scale"] = _VEL_TRACKING_TRANSITION_STD_SCALE
        self.rewards.track_ang_vel_z_exp.params["transition_duration_s"] = _VEL_TRACKING_TRANSITION_DURATION_S
        self.rewards.upward.weight = _UPWARD_TARGET_WEIGHT * reward_weight_scale

        # Walk These Ways terms start at 50% and ramp to 100% by iteration 1000 (see curriculum).
        wtw_target_weights = {
            "wtw_tracking_contacts_shaped_force": _WTW_TRACKING_FORCE_TARGET_WEIGHT,
            "wtw_tracking_contacts_shaped_vel": _WTW_TRACKING_VEL_TARGET_WEIGHT,
            "wtw_feet_clearance_cmd_linear": _WTW_FEET_CLEARANCE_TARGET_WEIGHT,
            "wtw_raibert_heuristic": _WTW_RAIBERT_TARGET_WEIGHT,
            "wtw_orientation_control": _WTW_ORIENTATION_TARGET_WEIGHT,
            "wtw_jump": _WTW_JUMP_TARGET_WEIGHT,
        }
        for term_name, target_weight in wtw_target_weights.items():
            getattr(self.rewards, term_name).weight = target_weight * wtw_weight_scale

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
        self.rewards.feet_lateral_regulation.weight = _FEET_LATERAL_REGULATION_TARGET_WEIGHT
        self.rewards.feet_lateral_regulation.params["asset_cfg"].body_names = [self.foot_link_name]

        # Penalize base motion on axes with zero velocity command (yaw weighted 5x heavier).
        self.rewards.uncommanded_base_vel.weight = _UNCOMMANDED_BASE_VEL_PENALTY_TARGET_WEIGHT

        # Penalize yaw spin when the yaw command is effectively zero; ramp 0 → target by iteration 1000.
        self.rewards.uncommanded_ang_vel_z.weight = _UNCOMMANDED_ANG_VEL_Z_TARGET_WEIGHT * ramp_from_zero_scale

        if not self.enable_reward_weight_curriculum:
            self.curriculum.track_lin_vel_xy_weight = None
            self.curriculum.track_ang_vel_z_weight = None
            self.curriculum.upward_weight = None
            self.curriculum.action_rate_l2_weight = None
            self.curriculum.action_smoothness_l2_weight = None
            self.curriculum.ang_vel_xy_l2_weight = None
            self.curriculum.wtw_tracking_force_weight = None
            self.curriculum.wtw_tracking_vel_weight = None
            self.curriculum.wtw_feet_clearance_weight = None
            self.curriculum.wtw_raibert_weight = None
            self.curriculum.wtw_orientation_weight = None
            self.curriculum.wtw_jump_weight = None
            self.curriculum.uncommanded_ang_vel_z_weight = None
            self.curriculum.calf_clearance_weight = None
            self.curriculum.footswing_height = None

        if self.__class__.__name__ == "UnitreeGo2WFlatLegsEnvCfg":
            self.disable_zero_weight_rewards()

    def disable_zero_weight_rewards(self):
        """Disable zero-weight rewards, but keep curriculum-ramped terms registered."""
        for attr in dir(self.rewards):
            if not attr.startswith("__") and attr not in _CURRICULUM_RAMPED_REWARD_TERM_NAMES:
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr is not None and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)
