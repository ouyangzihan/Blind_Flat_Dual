# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Elapsed time since the last command resample (used by velocity-tracking rewards).
        self.time_since_resample = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float):
        """Update commands and advance the post-resample timer."""
        super().compute(dt)
        self.time_since_resample += dt

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        self.time_since_resample[env_ids] = 0.0

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class UniformLateralYawVelocityCommand(UniformThresholdVelocityCommand):
    """Command generator that resamples until lateral or yaw velocity is non-zero."""

    cfg: UniformLateralYawVelocityCommandCfg  # type: ignore

    def _normalize_env_ids(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Convert env ids to a flat long tensor for consistent indexing."""
        if isinstance(env_ids, torch.Tensor):
            return env_ids.flatten().to(device=self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands until |vy| or |wz| exceeds the threshold."""
        pending_env_ids = self._normalize_env_ids(env_ids)
        if pending_env_ids.numel() == 0:
            return

        for _ in range(self.cfg.max_resample_attempts):
            super()._resample_command(pending_env_ids)
            vy = torch.abs(self.vel_command_b[pending_env_ids, 1])
            wz = torch.abs(self.vel_command_b[pending_env_ids, 2])
            valid_mask = (vy > self.cfg.lateral_yaw_threshold) | (wz > self.cfg.lateral_yaw_threshold)
            pending_env_ids = pending_env_ids[~valid_mask]
            if pending_env_ids.numel() == 0:
                return

        use_lateral = torch.rand(pending_env_ids.numel(), device=self.device) > 0.5
        lateral_sign = torch.where(torch.rand(pending_env_ids.numel(), device=self.device) > 0.5, 1.0, -1.0)
        yaw_sign = torch.where(torch.rand(pending_env_ids.numel(), device=self.device) > 0.5, 1.0, -1.0)
        lateral_cmd = lateral_sign * torch.clamp(
            torch.abs(self.vel_command_b[pending_env_ids, 1]),
            min=self.cfg.lateral_yaw_threshold,
        )
        yaw_cmd = yaw_sign * torch.clamp(
            torch.abs(self.vel_command_b[pending_env_ids, 2]),
            min=self.cfg.lateral_yaw_threshold,
        )
        self.vel_command_b[pending_env_ids, 1] = torch.where(use_lateral, lateral_cmd, 0.0)
        self.vel_command_b[pending_env_ids, 2] = torch.where(use_lateral, 0.0, yaw_cmd)


@configclass
class UniformLateralYawVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """Configuration for commands with at least one non-zero lateral or yaw component."""

    class_type: type = UniformLateralYawVelocityCommand

    lateral_yaw_threshold: float = 0.1
    max_resample_attempts: int = 100


class DualFinetuneVelocityCommand(UniformThresholdVelocityCommand):
    """Per-axis mixture velocity commands for wheels dual-finetune.

    ``vx`` and ``vy``/``yaw`` use different mixture probabilities (see cfg). Components
    with ``|value| <= deadzone`` are hard-set to 0. Legs-mode (``|vy|`` or ``|wz|``
    above deadzone) uses a short command lifetime; otherwise a longer wheels lifetime.
    When ``zero_lateral_yaw_on_episode_start`` is True, the first command of each
    episode forces ``vy=0`` and ``wz=0`` (wheels-only start).
    """

    cfg: DualFinetuneVelocityCommandCfg  # type: ignore

    def _normalize_env_ids(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.flatten().to(device=self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

    def _sample_axis(
        self,
        num: int,
        value_min: float,
        value_max: float,
        *,
        prob_zero: float,
        prob_max: float,
        prob_min: float,
    ) -> torch.Tensor:
        """Sample one command axis for ``num`` envs with the given mixture."""
        u = torch.rand(num, device=self.device)
        out = torch.empty(num, device=self.device)
        zero_mask = u < prob_zero
        out[zero_mask] = 0.0
        max_mask = (u >= prob_zero) & (u < prob_zero + prob_max)
        out[max_mask] = value_max
        min_lo = prob_zero + prob_max
        min_mask = (u >= min_lo) & (u < min_lo + prob_min)
        out[min_mask] = value_min
        rand_mask = u >= (min_lo + prob_min)
        if rand_mask.any():
            out[rand_mask] = torch.empty(int(rand_mask.sum().item()), device=self.device).uniform_(
                value_min, value_max
            )
        # Hard-zero near-zero values so they match play_dual deadzone.
        out = torch.where(out.abs() <= self.cfg.command_deadzone, torch.zeros_like(out), out)
        return out

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = self._normalize_env_ids(env_ids)
        if env_ids_t.numel() == 0:
            return

        n = env_ids_t.numel()
        # No explicit standing reservation: standing happens when all axes are zero.
        self.is_standing_env[env_ids_t] = False
        if self.cfg.heading_command:
            self.is_heading_env[env_ids_t] = False

        self.vel_command_b[env_ids_t, 0] = self._sample_axis(
            n,
            *self.cfg.ranges.lin_vel_x,
            prob_zero=self.cfg.vx_prob_zero,
            prob_max=self.cfg.vx_prob_max,
            prob_min=self.cfg.vx_prob_min,
        )
        self.vel_command_b[env_ids_t, 1] = self._sample_axis(
            n,
            *self.cfg.ranges.lin_vel_y,
            prob_zero=self.cfg.lateral_yaw_prob_zero,
            prob_max=self.cfg.lateral_yaw_prob_max,
            prob_min=self.cfg.lateral_yaw_prob_min,
        )
        self.vel_command_b[env_ids_t, 2] = self._sample_axis(
            n,
            *self.cfg.ranges.ang_vel_z,
            prob_zero=self.cfg.lateral_yaw_prob_zero,
            prob_max=self.cfg.lateral_yaw_prob_max,
            prob_min=self.cfg.lateral_yaw_prob_min,
        )

        # Episode reset zeros ``command_counter`` before ``_resample``; mid-episode
        # resamples already have counter >= 1. Force wheels-only (vy=wz=0) at start.
        if self.cfg.zero_lateral_yaw_on_episode_start:
            episode_start = self.command_counter[env_ids_t] == 0
            if episode_start.any():
                start_ids = env_ids_t[episode_start]
                self.vel_command_b[start_ids, 1] = 0.0
                self.vel_command_b[start_ids, 2] = 0.0

        # Legs-mode (nonzero vy or yaw after deadzone) → short lifetime; else wheels 2–3 s.
        vy = torch.abs(self.vel_command_b[env_ids_t, 1])
        wz = torch.abs(self.vel_command_b[env_ids_t, 2])
        is_legs = (vy > self.cfg.command_deadzone) | (wz > self.cfg.command_deadzone)
        time_left = torch.full(
            (n,),
            self.cfg.legs_command_duration_s,
            device=self.device,
            dtype=self.time_left.dtype,
        )
        wheels_mask = ~is_legs
        num_wheels = int(wheels_mask.sum().item())
        if num_wheels > 0:
            time_left[wheels_mask] = torch.empty(
                num_wheels, device=self.device, dtype=self.time_left.dtype
            ).uniform_(*self.cfg.wheels_command_duration_range)
        self.time_left[env_ids_t] = time_left


@configclass
class DualFinetuneVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """Configuration for dual-finetune per-axis mixture velocity commands."""

    class_type: type = DualFinetuneVelocityCommand

    rel_standing_envs: float = 0.0
    """Disabled; standing arises when vx=vy=yaw are all zero after deadzone."""

    # Forward velocity (vx): favor nonzero / endpoint commands for wheels training.
    vx_prob_zero: float = 0.25
    vx_prob_max: float = 0.30
    vx_prob_min: float = 0.30
    # Remaining mass (0.15) is uniform in range.

    # Lateral / yaw (vy, wz): mostly zero so wheels mode dominates; short legs bursts.
    lateral_yaw_prob_zero: float = 0.70
    lateral_yaw_prob_max: float = 0.12
    lateral_yaw_prob_min: float = 0.12
    # Remaining mass (0.10) is uniform in range.

    command_deadzone: float = 0.05
    """Hard-zero |component| at or below this; also defines legs-mode (|vy| or |wz| above)."""

    zero_lateral_yaw_on_episode_start: bool = True
    """If True, force ``vy=0`` and ``wz=0`` on the first command of each episode."""

    legs_command_duration_s: float = 1.0
    """Resample interval when |vy| or |wz| exceeds ``command_deadzone``."""

    wheels_command_duration_range: tuple[float, float] = (2.0, 3.0)
    """Resample interval range when the command is wheels-only or standing."""


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
