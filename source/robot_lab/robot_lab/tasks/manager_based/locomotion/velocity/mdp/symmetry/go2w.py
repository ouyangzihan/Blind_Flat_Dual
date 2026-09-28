# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Left-right symmetry augmentation for Unitree Go2W velocity locomotion."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["Go2WSymmetryMapper", "compute_symmetric_states"]

# Leg joint order: FR, FL, RR, RL (3 DoF each).
_LEG_JOINT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
_LEG_JOINT_SIGN = [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]
# Wheel joint order: FR_foot, FL_foot, RR_foot, RL_foot.
_WHEEL_PERM = [1, 0, 3, 2]
_WHEEL_SIGN = [-1, -1, -1, -1]
_FOOT_PERM = [1, 0, 3, 2]
_VECTOR_SIGNS = {
    "base_lin_vel": [1, -1, 1],
    "base_ang_vel": [-1, 1, -1],
    "projected_gravity": [1, -1, 1],
    "velocity_commands": [1, -1, -1],
}
# Go2W joint_pos uses joint_pos_rel_without_wheel: 16-dim vector with wheel slots zeroed.
_JOINT_TERMS = {"joint_pos", "joint_vel", "joint_acc", "joint_torque", "actions"}


class Go2WSymmetryMapper:
    """Mirror Go2W observations and actions across the robot sagittal plane."""

    def __init__(self, env) -> None:
        self.env = env

    @property
    def base_env(self):
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

    def reverse_vector(self, value: torch.Tensor, signs: list[int]) -> torch.Tensor:
        return value * value.new_tensor(signs)

    def reverse_legs(self, value: torch.Tensor) -> torch.Tensor:
        perm = torch.tensor(_LEG_JOINT_PERM, device=value.device)
        signs = value.new_tensor(_LEG_JOINT_SIGN)
        return value.index_select(-1, perm) * signs

    def permute_legs(self, value: torch.Tensor) -> torch.Tensor:
        perm = torch.tensor(_LEG_JOINT_PERM, device=value.device)
        return value.index_select(-1, perm)

    def reverse_wheels(self, value: torch.Tensor) -> torch.Tensor:
        perm = torch.tensor(_WHEEL_PERM, device=value.device)
        signs = value.new_tensor(_WHEEL_SIGN)
        return value.index_select(-1, perm) * signs

    def permute_wheels(self, value: torch.Tensor) -> torch.Tensor:
        perm = torch.tensor(_WHEEL_PERM, device=value.device)
        return value.index_select(-1, perm)

    def reverse_full_joints(self, value: torch.Tensor) -> torch.Tensor:
        legs = self.reverse_legs(value[..., :12])
        wheels = self.reverse_wheels(value[..., 12:])
        return torch.cat([legs, wheels], dim=-1)

    def permute_full_joints(self, value: torch.Tensor) -> torch.Tensor:
        legs = self.permute_legs(value[..., :12])
        wheels = self.permute_wheels(value[..., 12:])
        return torch.cat([legs, wheels], dim=-1)

    def restore_history(self, value: torch.Tensor, cfg) -> torch.Tensor:
        history_length = getattr(cfg, "history_length", 0)
        if history_length > 0 and getattr(cfg, "flatten_history_dim", False):
            term_dim = value.shape[-1] // history_length
            return value.reshape(*value.shape[:-1], history_length, term_dim)
        return value

    def flatten_history(self, value: torch.Tensor, cfg) -> torch.Tensor:
        history_length = getattr(cfg, "history_length", 0)
        if history_length > 0 and getattr(cfg, "flatten_history_dim", False):
            return value.reshape(*value.shape[:-2], -1)
        return value

    def reverse_feet(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] % len(_FOOT_PERM) != 0:
            return value
        perm = torch.tensor(_FOOT_PERM, device=value.device)
        return value.reshape(*value.shape[:-1], -1, len(_FOOT_PERM)).index_select(-1, perm).reshape_as(value)

    def height_scan_shape(self, cfg) -> tuple[int, int, str] | None:
        sensor_cfg = cfg.params.get("sensor_cfg")
        scene_cfg = getattr(self.base_env.cfg, "scene", None)
        sensor_scene_cfg = getattr(scene_cfg, sensor_cfg.name, None) if sensor_cfg is not None else None
        pattern_cfg = getattr(sensor_scene_cfg, "pattern_cfg", None)
        if pattern_cfg is None:
            return None

        nx = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        ny = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        return nx, ny, pattern_cfg.ordering

    def reverse_height_scan(self, cfg, value: torch.Tensor) -> torch.Tensor:
        shape = self.height_scan_shape(cfg)
        if shape is None:
            return value

        nx, ny, ordering = shape
        if value.shape[-1] != nx * ny:
            return value

        if ordering == "xy":
            return value.reshape(*value.shape[:-1], ny, nx).flip(-2).reshape_as(value)
        return value.reshape(*value.shape[:-1], nx, ny).flip(-1).reshape_as(value)

    def reverse_joint_term(self, value: torch.Tensor) -> torch.Tensor:
        width = value.shape[-1]
        if width == 12:
            return self.reverse_legs(value)
        if width == 16:
            return self.reverse_full_joints(value)
        return value

    def reverse_term(self, name: str, cfg, value: torch.Tensor) -> torch.Tensor:
        if name in _VECTOR_SIGNS:
            return self.reverse_vector(value, _VECTOR_SIGNS[name])
        if name in _JOINT_TERMS:
            return self.reverse_joint_term(value)
        if name == "contact_force":
            return self.reverse_feet(value)
        if name == "gait_clock":
            return self.reverse_feet(value)
        if name == "height_scan":
            return self.reverse_height_scan(cfg, value)
        return value

    def reverse_obs_group(self, group_name: str, value: torch.Tensor) -> torch.Tensor:
        manager = self.base_env.observation_manager
        names = manager._group_obs_term_names[group_name]
        dims = manager._group_obs_term_dim[group_name]
        cfgs = manager._group_obs_term_cfgs[group_name]
        widths = [int(math.prod(dim)) for dim in dims]
        if sum(widths) != value.shape[-1]:
            return value

        chunks = torch.split(value, widths, dim=-1)
        reversed_chunks = []
        for name, cfg, chunk in zip(names, cfgs, chunks):
            term = self.restore_history(chunk, cfg)
            term = self.reverse_term(name, cfg, term)
            reversed_chunks.append(self.flatten_history(term, cfg))
        return torch.cat(reversed_chunks, dim=-1)

    def reverse_obs(self, obs: TensorDict) -> TensorDict:
        manager = self.base_env.observation_manager
        reversed_obs = obs.clone()
        for key, value in obs.items():
            if key in manager._group_obs_term_names:
                reversed_obs[key] = self.reverse_obs_group(key, value)
        return reversed_obs

    def reverse_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] == 12:
            return self.reverse_legs(actions)
        if actions.shape[-1] == 16:
            return self.reverse_full_joints(actions)
        return actions

    def data_augmentation(self, obs: TensorDict, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        return (
            torch.cat([obs, self.reverse_obs(obs)], dim=0),
            torch.cat([actions, self.reverse_actions(actions)], dim=0),
        )


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations and actions with a left-right mirror (2x batch size).

    Mirrors the Go2 MoE-CTS symmetry scheme: original samples followed by their
    sagittal-plane reflections. Compatible with ``RslRlSymmetryCfg.use_data_augmentation``.
    """
    mapper = Go2WSymmetryMapper(env)

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug[:batch_size] = obs[:]
        for key, value in obs.items():
            if key in mapper.base_env.observation_manager._group_obs_term_names:
                obs_aug[key][batch_size:] = mapper.reverse_obs_group(key, value)
    else:
        obs_aug = None

    if actions is not None:
        actions_aug = torch.cat([actions, mapper.reverse_actions(actions)], dim=0)
    else:
        actions_aug = None

    return obs_aug, actions_aug
