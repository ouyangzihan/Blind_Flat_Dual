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
    """

    cfg: GaitBehaviorCommandCfg

    def __init__(self, cfg: GaitBehaviorCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 11, device=self.device)
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.foot_indices = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, device=self.device)
        self._smoothing_cdf = Normal(0.0, self.cfg.kappa_gait_probs).cdf

    @property
    def command(self) -> torch.Tensor:
        return self._command

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
        self._command[env_ids_t, 1] = sample_uniform(
            ranges.frequency[0], ranges.frequency[1], (len(env_ids),), device=self.device
        )
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

    def _update_command(self):
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


@configclass
class GaitBehaviorCommandCfg(CommandTermCfg):
    """Configuration for the Walk These Ways gait behavior command generator."""

    class_type: type = GaitBehaviorCommand

    asset_name: str = "robot"
    kappa_gait_probs: float = 0.07

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
