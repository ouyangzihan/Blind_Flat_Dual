# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


def reward_weight_linear_decay(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    target_weight: float,
    start_scale: float = 1.5,
    num_iterations: int = 500,
    steps_per_iteration: int = 24,
    second_decay_start_iteration: int | None = None,
    second_decay_iterations: int = 1000,
    second_end_scale: float = 0.5,
) -> float:
    """Linearly decay a reward term weight from ``start_scale * target_weight`` to ``target_weight``.

    The schedule is indexed by PPO learning iterations via ``env.common_step_counter`` and
    ``steps_per_iteration`` (typically ``num_steps_per_env`` from the runner config).

    If ``second_decay_start_iteration`` is set, the weight holds at ``target_weight`` until
    that iteration, then linearly decays to ``second_end_scale * target_weight`` over
    ``second_decay_iterations``.
    """
    del env_ids  # unused; curriculum is shared across all parallel environments

    num_steps = num_iterations * steps_per_iteration
    if env.common_step_counter >= num_steps:
        scale = 1.0
    else:
        progress = env.common_step_counter / num_steps
        scale = start_scale + (1.0 - start_scale) * progress

    if second_decay_start_iteration is not None:
        second_start_steps = second_decay_start_iteration * steps_per_iteration
        second_decay_steps = second_decay_iterations * steps_per_iteration
        if env.common_step_counter >= second_start_steps + second_decay_steps:
            scale = second_end_scale
        elif env.common_step_counter >= second_start_steps:
            progress = (env.common_step_counter - second_start_steps) / second_decay_steps
            scale = 1.0 + (second_end_scale - 1.0) * progress

    weight = target_weight * scale
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if term_cfg.weight != weight:
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return weight


def reward_weight_linear_ramp(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    target_weight: float,
    start_scale: float = 0.0,
    num_iterations: int = 500,
    steps_per_iteration: int = 24,
) -> float:
    """Linearly ramp a reward term weight from ``start_scale * target_weight`` to ``target_weight``.

    The schedule is indexed by PPO learning iterations via ``env.common_step_counter`` and
    ``steps_per_iteration`` (typically ``num_steps_per_env`` from the runner config).
    """
    del env_ids  # unused; curriculum is shared across all parallel environments

    num_steps = num_iterations * steps_per_iteration
    if env.common_step_counter >= num_steps:
        scale = 1.0
    else:
        progress = env.common_step_counter / num_steps
        scale = start_scale + (1.0 - start_scale) * progress

    weight = target_weight * scale
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if term_cfg.weight != weight:
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return weight


def footswing_height_linear_ramp(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "gait_behavior",
    target_range: tuple[float, float] = (0.05, 0.05),
    start_scale: float = 0.2,
    num_iterations: int = 1000,
    steps_per_iteration: int = 24,
) -> float:
    """Linearly ramp gait ``footswing_height`` from ``start_scale * target`` to ``target``.

    Updates ``command_manager`` range and live ``_nominal_footswing`` so the clearance
    target tracks the schedule immediately (not only on the next command resample).
    """
    del env_ids  # unused; curriculum is shared across all parallel environments

    num_steps = num_iterations * steps_per_iteration
    if env.common_step_counter >= num_steps:
        scale = 1.0
    else:
        progress = env.common_step_counter / num_steps
        scale = start_scale + (1.0 - start_scale) * progress

    new_lo = target_range[0] * scale
    new_hi = target_range[1] * scale
    gait_cmd = env.command_manager.get_term(command_name)
    ranges = gait_cmd.cfg.ranges
    cur = ranges.footswing_height
    if cur[0] != new_lo or cur[1] != new_hi:
        ranges.footswing_height = (new_lo, new_hi)
        if hasattr(gait_cmd, "_nominal_footswing"):
            if abs(new_hi - new_lo) < 1.0e-12:
                gait_cmd._nominal_footswing[:] = new_lo
            else:
                # Preserve relative position in the previous range when both ends move.
                old_span = cur[1] - cur[0]
                if abs(old_span) < 1.0e-12:
                    frac = 0.5
                else:
                    frac = ((gait_cmd._nominal_footswing - cur[0]) / old_span).clamp(0.0, 1.0)
                gait_cmd._nominal_footswing[:] = new_lo + frac * (new_hi - new_lo)
            if hasattr(gait_cmd, "_sync_velocity_modulated_params"):
                gait_cmd._sync_velocity_modulated_params()

    return new_hi
