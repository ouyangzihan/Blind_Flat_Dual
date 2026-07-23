# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import (
    UnitreeGo2WRewardsCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg

from .flat_env_cfg import UnitreeGo2WFlatEnvCfg

_TRACK_LIN_VEL_X_TARGET_WEIGHT = 6.0
_UPWARD_TARGET_WEIGHT = 2.0
_REWARD_WEIGHT_START_SCALE = 2.0
_REWARD_WEIGHT_DECAY_ITERATIONS = 500
_REWARD_WEIGHT_STEPS_PER_ITERATION = 24  # matches UnitreeGo2WFlatWheelsPPORunnerCfg.num_steps_per_env


@configclass
class UnitreeGo2WFlatWheelsRewardsCfg(UnitreeGo2WRewardsCfg):
    """Reward terms for x-direction wheel-dominant locomotion."""

    ang_vel_x_l2 = RewTerm(func=mdp.ang_vel_x_l2, weight=0.0)
    ang_vel_y_l2 = RewTerm(func=mdp.ang_vel_y_l2, weight=0.0)

    track_lin_vel_x_exp = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    run_still = RewTerm(
        func=mdp.run_still,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    similar_to_default = RewTerm(
        func=mdp.similar_to_default,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    wheels_planted = RewTerm(
        func=mdp.wheels_planted,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "force_threshold": 1.0,
        },
    )
    hip_joint_default = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    body_height_default = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": None,
            "target_height": 0.0,
        },
    )
    hip_pos_imbalance = RewTerm(
        func=mdp.joint_pos_imbalance,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"]],
        },
    )
    thigh_pos_imbalance = RewTerm(
        func=mdp.joint_pos_imbalance,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"]],
        },
    )
    calf_pos_imbalance = RewTerm(
        func=mdp.joint_pos_imbalance,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"]],
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
class UnitreeGo2WFlatWheelsCurriculumCfg(CurriculumCfg):
    """Curriculum for flat-wheels training; inherits disabled terrain curriculum from flat env."""

    track_lin_vel_x_weight = CurrTerm(
        func=mdp.reward_weight_linear_decay,
        params={
            "term_name": "track_lin_vel_x_exp",
            "target_weight": _TRACK_LIN_VEL_X_TARGET_WEIGHT,
            "start_scale": _REWARD_WEIGHT_START_SCALE,
            "num_iterations": _REWARD_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _REWARD_WEIGHT_STEPS_PER_ITERATION,
        },
    )
    upward_weight = CurrTerm(
        func=mdp.reward_weight_linear_decay,
        params={
            "term_name": "upward",
            "target_weight": _UPWARD_TARGET_WEIGHT,
            "start_scale": _REWARD_WEIGHT_START_SCALE,
            "num_iterations": _REWARD_WEIGHT_DECAY_ITERATIONS,
            "steps_per_iteration": _REWARD_WEIGHT_STEPS_PER_ITERATION,
        },
    )


@configclass
class UnitreeGo2WFlatWheelsEnvCfg(UnitreeGo2WFlatEnvCfg):
    """Flat terrain env for Go2W with x-only velocity commands and wheel-dominant rewards."""

    rewards: UnitreeGo2WFlatWheelsRewardsCfg = UnitreeGo2WFlatWheelsRewardsCfg()
    curriculum: UnitreeGo2WFlatWheelsCurriculumCfg = UnitreeGo2WFlatWheelsCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (3.9, 4.1)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.commands.base_velocity.rel_standing_envs = 0.2

        # ------------------------------Events------------------------------
        # Linear velocity push along base x/y/z (additive via push_by_setting_velocity).
        self.events.randomize_push_robot.interval_range_s = (1.9, 2.1)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-1., 1.),
            "y": (-1., 1.),
            "z": (-1., 1.),
        }
        # Angular velocity push about base roll/pitch/yaw (x/y/z).
        self.events.randomize_push_robot_ang_vel = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.9, 3.1),
            params={
                "velocity_range": {
                    "roll": (-1., 1.),
                    "pitch": (-1., 1.),
                    "yaw": (-1., 1.),
                }
            },
        )

        # ------------------------------Rewards------------------------------
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0

        # Selected rewards start at 200% and linearly decay to 100% by iteration 500 (see curriculum).
        self.rewards.track_lin_vel_x_exp.weight = _TRACK_LIN_VEL_X_TARGET_WEIGHT * _REWARD_WEIGHT_START_SCALE
        self.rewards.lin_vel_y_l2.weight = -10.0 # -3.0
        self.rewards.ang_vel_z_l2.weight = -5.0  # -1.5
        self.rewards.upward.weight = _UPWARD_TARGET_WEIGHT * _REWARD_WEIGHT_START_SCALE

        self.rewards.lin_acc_z_l2.weight = -.001
        self.rewards.lin_acc_z_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.ang_acc_xy_l2.weight = -0.0000125
        self.rewards.ang_acc_xy_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.ang_vel_xy_l2.weight = 0
        self.rewards.ang_vel_x_l2.weight = -0.1  # 2x inherited ang_vel_xy_l2 weight (-0.05)
        self.rewards.ang_vel_y_l2.weight = -0.025  # 0.5x inherited ang_vel_xy_l2 weight (-0.05)

        self.rewards.run_still.weight = -.0002 # -1.0
        self.rewards.run_still.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.similar_to_default.weight = -.16 # -2.0
        self.rewards.similar_to_default.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.wheels_planted.weight = -.20 # -1.0
        self.rewards.wheels_planted.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.hip_joint_default.weight = -0.2
        self.rewards.hip_joint_default.params["asset_cfg"].joint_names = [".*_hip_joint"]

        self.rewards.body_height_default.weight = -1.0
        self.rewards.body_height_default.params["target_height"] = 0.38
        self.rewards.body_height_default.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.stand_still.weight = -0.3
        self.rewards.joint_pos_penalty.weight = -0.7

        self.rewards.hip_pos_imbalance.weight = -60
        self.rewards.thigh_pos_imbalance.weight = -60
        self.rewards.calf_pos_imbalance.weight = -60

        self.rewards.joint_acc_l2.weight = -5.0e-7

        if self.__class__.__name__ == "UnitreeGo2WFlatWheelsEnvCfg":
            self.disable_zero_weight_rewards()
