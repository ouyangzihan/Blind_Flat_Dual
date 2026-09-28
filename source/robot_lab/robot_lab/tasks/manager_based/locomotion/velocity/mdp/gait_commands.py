# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Gait behavior command generator from Walk These Ways (MoB)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch.distributions import Normal

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GaitBehaviorCommand(CommandTerm):
    """Behavior command generator for Walk These Ways augmented auxiliary rewards.

    Command layout (11 dims):
        0: body height offset
        1: gait frequency (Hz)
        2: gait phase
        3: gait offset
        4: gait bound
        5: stance duration fraction
        6: footswing height
        7: body pitch
        8: body roll
        9: stance width
        10: stance length

    When ``velocity_command_name`` is set, frequency tracks ``|vy|`` and ``|yaw|``
    each step: ``freq = frequency_base * (1 + |vy| * frequency_vy_scale + |yaw| *
    frequency_yaw_scale)``. ``frequency_base`` may be a scalar or a ``(lo, hi)``
    range resampled per env. Yaw is commanded angular velocity in rad/s.

    When ``enable_planted_gait`` is set and ``|vy|`` and ``|yaw|`` are both below
    ``cmd_threshold``, footswing height is forced to 0 and desired contacts are
    all stance so WTW contact/clearance terms keep wheels on the ground.
    """

    cfg: GaitBehaviorCommandCfg

    def __init__(self, cfg: GaitBehaviorCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 11, device=self.device)
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.foot_indices = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, device=self.device)
        # Resampled / curriculum footswing; restored when leaving planted mode.
        self._nominal_footswing = torch.zeros(self.num_envs, device=self.device)
        # Per-env gait frequency at |vy|=0, |yaw|=0; resampled with the command.
        freq_lo, freq_hi = _frequency_base_range(cfg)
        self._frequency_base = torch.full((self.num_envs,), 0.5 * (freq_lo + freq_hi), device=self.device)
        self.is_planted_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._smoothing_cdf = Normal(0.0, self.cfg.kappa_gait_probs).cdf

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def frequency_base(self) -> torch.Tensor:
        """Per-env gait frequency at ``|vy|=0``, ``|yaw|=0`` (Hz)."""
        return self._frequency_base

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        ranges = self.cfg.ranges
        self._command[env_ids_t, 0] = sample_uniform(
            ranges.body_height[0], ranges.body_height[1], (len(env_ids),), device=self.device
        )
        freq_lo, freq_hi = _frequency_base_range(self.cfg)
        self._frequency_base[env_ids_t] = sample_uniform(
            freq_lo, freq_hi, (len(env_ids),), device=self.device
        )
        self._command[env_ids_t, 1] = self._frequency_base[env_ids_t]
        self._command[env_ids_t, 2] = sample_uniform(
            ranges.phase[0], ranges.phase[1], (len(env_ids),), device=self.device
        )
        self._command[env_ids_t, 3] = sample_uniform(
            ranges.offset[0], ranges.offset[1], (len(env_ids),), device=self.device
        )
        self._command[env_ids_t, 4] = sample_uniform(
            ranges.bound[0], ranges.bound[1], (len(env_ids),), device=self.device
        )
        self._command[env_ids_t, 5] = sample_uniform(
            ranges.duration[0], ranges.duration[1], (len(env_ids),), device=self.device
        )
        self._command[env_ids_t, 6] = sample_uniform(
            ranges.footswing_height[0], ranges.footswing_height[1], (len(env_ids),), device=self.device
        )
        self._nominal_footswing[env_ids_t] = self._command[env_ids_t, 6]
        self._command[env_ids_t, 7] = sample_uniform(
            ranges.body_pitch[0], ranges.body_pitch[1], (len(env_ids),), device=self.device
        )
        self._command[env_ids_t, 8] = sample_uniform(
            ranges.body_roll[0], ranges.body_roll[1], (len(env_ids),), device=self.device
        )
        self._command[env_ids_t, 9] = sample_uniform(
            ranges.stance_width[0], ranges.stance_width[1], (len(env_ids),), device=self.device
        )
        self._command[env_ids_t, 10] = sample_uniform(
            ranges.stance_length[0], ranges.stance_length[1], (len(env_ids),), device=self.device
        )
        self.gait_indices[env_ids_t] = 0.0

    def _sync_velocity_modulated_params(self):
        """Update frequency / footswing / planted flag from the velocity command."""
        self._command[:, 6] = self._nominal_footswing
        self.is_planted_env[:] = False
        if self.cfg.velocity_command_name is None:
            return

        vel_cmd = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        vy_abs = torch.abs(vel_cmd[:, 1])
        yaw_abs = torch.abs(vel_cmd[:, 2])

        # freq = base * (1 + |vy| * vy_scale + |yaw| * yaw_scale); base is resampled per env.
        self._command[:, 1] = self._frequency_base * (
            1 + vy_abs * self.cfg.frequency_vy_scale + yaw_abs * self.cfg.frequency_yaw_scale
        )

        if not self.cfg.enable_planted_gait:
            return

        planted = (vy_abs < self.cfg.cmd_threshold) & (yaw_abs < self.cfg.cmd_threshold)
        self.is_planted_env[:] = planted
        self._command[:, 6] = torch.where(planted, torch.zeros_like(self._nominal_footswing), self._nominal_footswing)

    def _update_command(self):
        self._sync_velocity_modulated_params()

        frequencies = self._command[:, 1]
        phases = self._command[:, 2]
        offsets = self._command[:, 3]
        bounds = self._command[:, 4]
        durations = self._command[:, 5]

        self.gait_indices = torch.remainder(self.gait_indices + self._env.step_dt * frequencies, 1.0)

        foot_phase_offsets = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]

        self.foot_indices = torch.remainder(
            torch.stack(foot_phase_offsets, dim=1),
            1.0,
        )

        for idx in range(4):
            foot_phase = foot_phase_offsets[idx]
            stance_mask = torch.remainder(foot_phase, 1.0) < durations
            swing_mask = torch.remainder(foot_phase, 1.0) > durations
            foot_phase = foot_phase.clone()
            foot_phase[stance_mask] = torch.remainder(foot_phase[stance_mask], 1.0) * (0.5 / durations[stance_mask])
            foot_phase[swing_mask] = 0.5 + (torch.remainder(foot_phase[swing_mask], 1.0) - durations[swing_mask]) * (
                0.5 / (1.0 - durations[swing_mask])
            )
            foot_phase_offsets[idx] = foot_phase

        smoothing = []
        for idx in range(4):
            phase = torch.remainder(foot_phase_offsets[idx], 1.0)
            smoothing.append(
                self._smoothing_cdf(phase) * (1.0 - self._smoothing_cdf(phase - 0.5))
                + self._smoothing_cdf(phase - 1.0) * (1.0 - self._smoothing_cdf(phase - 0.5 - 1.0))
            )
        self.desired_contact_states = torch.stack(smoothing, dim=1)

        if self.cfg.enable_planted_gait and self.is_planted_env.any():
            self.desired_contact_states[self.is_planted_env] = 1.0


def _frequency_base_range(cfg: GaitBehaviorCommandCfg) -> tuple[float, float]:
    """Return ``(lo, hi)`` for sampling per-env ``frequency_base``."""
    freq_base = cfg.frequency_base
    if freq_base is None:
        return cfg.ranges.frequency
    if isinstance(freq_base, (tuple, list)):
        return (float(freq_base[0]), float(freq_base[1]))
    return (float(freq_base), float(freq_base))


@configclass
class GaitBehaviorCommandCfg(CommandTermCfg):
    """Configuration for the Walk These Ways gait behavior command generator."""

    class_type: type = GaitBehaviorCommand

    asset_name: str = "robot"
    kappa_gait_probs: float = 0.07
    # When set, frequency tracks |vy| and |yaw| (rad/s) each step.
    velocity_command_name: str | None = None
    # Scalar, ``(lo, hi)`` range, or None → ``ranges.frequency``. Resampled per env.
    frequency_base: float | tuple[float, float] | None = None
    frequency_vy_scale: float = 0.5
    frequency_yaw_scale: float = 0.5
    # When True, |vy| and |yaw| below cmd_threshold force footswing=0 and all-stance.
    enable_planted_gait: bool = False
    cmd_threshold: float = 0.05

    @configclass
    class Ranges:
        """Behavior parameter sampling ranges."""

        body_height: tuple[float, float] = (-0.05, 0.05)
        frequency: tuple[float, float] = (2.0, 2.0)
        phase: tuple[float, float] = (0.0, 0.0)
        offset: tuple[float, float] = (0.0, 0.0)
        bound: tuple[float, float] = (0.0, 0.0)
        duration: tuple[float, float] = (0.5, 0.5)
        footswing_height: tuple[float, float] = (0.06, 0.09)
        body_pitch: tuple[float, float] = (-0.1, 0.1)
        body_roll: tuple[float, float] = (-0.1, 0.1)
        stance_width: tuple[float, float] = (0.25, 0.35)
        stance_length: tuple[float, float] = (0.38, 0.48)

    ranges: Ranges = Ranges()
