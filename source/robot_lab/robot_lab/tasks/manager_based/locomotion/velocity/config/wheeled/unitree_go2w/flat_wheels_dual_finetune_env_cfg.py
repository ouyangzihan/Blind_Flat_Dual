# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg

from .flat_legs_env_cfg import _CMD_NEAR_ZERO_THRESHOLD, _FOOTSWING_HEIGHT_TARGET
from .flat_wheels_env_cfg import UnitreeGo2WFlatWheelsEnvCfg, UnitreeGo2WFlatWheelsRewardsCfg


@configclass
class UnitreeGo2WFlatWheelsDualFinetuneCommandsCfg(CommandsCfg):
    """Per-axis mixture commands; legs-mode segments last 1 s, otherwise 2–3 s."""

    base_velocity = mdp.DualFinetuneVelocityCommandCfg(
        asset_name="robot",
        # Overwritten per sample in DualFinetuneVelocityCommand based on legs vs wheels.
        resampling_time_range=(6.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        legs_command_duration_s=1.0,
        wheels_command_duration_range=(6.0, 8.0),
        command_deadzone=0.05,
        # Episode starts in wheels mode; mid-episode resampling can still sample vy/yaw.
        zero_lateral_yaw_on_episode_start=True,
        ranges=mdp.DualFinetuneVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.75, 0.75),
            ang_vel_z=(-1.5, 1.5),
            heading=(-math.pi, math.pi),
        ),
    )
    # Clock for the frozen legs policy (wheels policy obs stay 57-dim and do not read this).
    gait_behavior = mdp.GaitBehaviorCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        kappa_gait_probs=0.07,
        velocity_command_name="base_velocity",
        frequency_base=(1.0, 1.5),
        frequency_vy_scale=0.5,
        frequency_yaw_scale=0.25,
        cmd_threshold=_CMD_NEAR_ZERO_THRESHOLD,
        enable_planted_gait=True,
        ranges=mdp.GaitBehaviorCommandCfg.Ranges(
            body_height=(0.0, 0.0),
            frequency=(1.0, 1.5),
            phase=(0.5, 0.5),
            offset=(0.0, 0.0),
            bound=(0.0, 0.0),
            duration=(0.5, 0.5),
            footswing_height=_FOOTSWING_HEIGHT_TARGET,
            body_pitch=(0.0, 0.0),
            body_roll=(0.0, 0.0),
            stance_width=(0.284, 0.284),
            stance_length=(0.387, 0.387),
        ),
    )


@configclass
class UnitreeGo2WFlatWheelsDualFinetuneRewardsCfg(UnitreeGo2WFlatWheelsRewardsCfg):
    """Wheels rewards plus zero-vx wheel-speed penalty for dual finetune."""

    wheel_speed_zero_vx = RewTerm(
        func=mdp.wheel_speed_zero_vx_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "vx_threshold": 0.05,
        },
    )


@configclass
class UnitreeGo2WFlatWheelsDualFinetuneEnvCfg(UnitreeGo2WFlatWheelsEnvCfg):
    """Wheels rewards + mixed dual-mode commands; no mid-episode push randomizations.

    Used by ``train_dual_finetune.py``: frozen legs policy acts in legs-mode, wheels
    policy is trained only on wheels-active / standing transitions (including handoffs).

    Set ``train_from_scratch=True`` (or pass ``--from_scratch``) to skip the pretrained
    wheels checkpoint and use flat-wheels learning rate / entropy instead of the
    conservative finetune values.

    ``reload_legs_interval`` (default 500) reloads the newest legs checkpoint from
    ``logs/rsl_rl/<legs_experiment>`` while a concurrent legs training job is writing
    checkpoints. Set to 0 (or ``--reload_legs_interval 0``) to freeze the initial legs
    policy. Pin the live legs run with ``--legs_load_run``.
    """

    commands: UnitreeGo2WFlatWheelsDualFinetuneCommandsCfg = UnitreeGo2WFlatWheelsDualFinetuneCommandsCfg()
    rewards: UnitreeGo2WFlatWheelsDualFinetuneRewardsCfg = UnitreeGo2WFlatWheelsDualFinetuneRewardsCfg()
    # When True, train_dual_finetune.py initializes the wheels policy randomly and
    # applies UnitreeGo2WFlatWheelsPPORunnerCfg rate / entropy_coef.
    train_from_scratch: bool = False
    # Dual-finetune runner: reload newest flat-legs checkpoint every N iterations (0 = off).
    reload_legs_interval: int = 500
    # After each legs→wheels handoff, scale these reward terms by this for this many sim seconds.
    action_rate_handoff_boost_duration_s: float = 0.75
    action_rate_handoff_boost_scale: float = 1.5
    action_rate_handoff_boost_terms: tuple[str, ...] = (
        "action_rate_l2",
        "lin_vel_y_l2",
        "ang_vel_z_l2",
        "lin_acc_z_l2",
        "ang_acc_xy_l2",
        "ang_acc_z_l2",
        "ang_vel_x_l2",
        "ang_vel_y_l2",
    )

    def __post_init__(self):
        super().__post_init__()

        # Dual-finetune relies on command-driven mode switches, not velocity/joint pushes.
        self.events.randomize_push_robot = None
        self.events.randomize_push_robot_ang_vel = None
        self.events.randomize_push_standing_leg_joints = None
        self.events.enable_standing_joint_push_reward_skip = None

        # Longer episodes so each env sees multiple legs→wheels handoffs.
        # self.episode_length_s = 25.0

        # Parent wheels cfg zeros vy/wz; restore dual-finetune per-axis mixture ranges.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (6.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.legs_command_duration_s = 1.0
        self.commands.base_velocity.wheels_command_duration_range = (6.0, 8.0)
        self.commands.base_velocity.command_deadzone = 0.05
        self.commands.base_velocity.zero_lateral_yaw_on_episode_start = True

        # Penalize spinning wheels when |vx| is below command deadzone.
        self.rewards.wheel_speed_zero_vx.weight = -0.0 # -0.01
        self.rewards.wheel_speed_zero_vx.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_speed_zero_vx.params["vx_threshold"] = 0.05

        # Listed terms stay at parent weights; DualFinetuneOnPolicyRunner scales them by
        # action_rate_handoff_boost_scale for action_rate_handoff_boost_duration_s after each
        # legs→wheels handoff (timer resets on every handoff).

        if self.__class__.__name__ == "UnitreeGo2WFlatWheelsDualFinetuneEnvCfg":
            self.disable_zero_weight_rewards()
