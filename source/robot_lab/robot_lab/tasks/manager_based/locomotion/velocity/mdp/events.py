# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .utils import is_env_assigned_to_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization,
            or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


# Per-env common_step_counter recorded when standing joint pushes are applied.
# Used by rewards to ignore one-step teleport artifacts (interval events run after rewards).
STANDING_JOINT_PUSH_STEP_ATTR = "_standing_joint_push_common_step"


def _ensure_standing_joint_push_step_buf(env: ManagerBasedEnv) -> torch.Tensor:
    """Allocate (if needed) and return the standing-joint-push step buffer."""
    num_envs = env.scene.num_envs
    buf = getattr(env, STANDING_JOINT_PUSH_STEP_ATTR, None)
    if buf is None or buf.shape[0] != num_envs:
        buf = torch.full((num_envs,), -1_000_000_000, device=env.device, dtype=torch.long)
        setattr(env, STANDING_JOINT_PUSH_STEP_ATTR, buf)
    return buf


def record_standing_joint_push(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Record that a standing joint push was applied at the current common step."""
    buf = _ensure_standing_joint_push_step_buf(env)
    buf[env_ids] = int(env.common_step_counter)


def standing_joint_push_exclude_mask(env: ManagerBasedEnv, exclude_steps: int = 1) -> torch.Tensor:
    """Return True for envs whose standing joint push should be excluded from rewards.

    Interval events run after reward computation, so artifacts appear on the following
    control step(s). With ``exclude_steps=1``, only the immediate next reward step is masked.
    """
    buf = getattr(env, STANDING_JOINT_PUSH_STEP_ATTR, None)
    if buf is None:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    steps_since = env.common_step_counter - buf
    return (steps_since >= 1) & (steps_since <= exclude_steps)


def enable_standing_joint_push_reward_skip(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    exclude_steps: int = 1,
):
    """Wrap ``reward_manager.compute`` to zero all rewards on standing-joint-push skip frames.

    Intended to be registered as a ``startup`` event. Subtracts the skipped step from
    per-term episode sums and clears ``_step_reward`` so logging matches the returned total.
    """
    del env_ids  # unused; wrap applies to all envs
    if getattr(env, "_standing_joint_push_reward_skip_installed", False):
        return

    reward_manager = env.reward_manager
    original_compute = reward_manager.compute

    def compute_with_skip(dt: float) -> torch.Tensor:
        reward_buf = original_compute(dt)
        mask = standing_joint_push_exclude_mask(env, exclude_steps)
        if not torch.any(mask):
            return reward_buf
        for term_idx, name in enumerate(reward_manager._term_names):
            term_value = reward_manager._step_reward[:, term_idx] * dt
            reward_manager._episode_sums[name][mask] -= term_value[mask]
            reward_manager._step_reward[mask, term_idx] = 0.0
        reward_buf[mask] = 0.0
        return reward_buf

    reward_manager.compute = compute_with_skip  # type: ignore[method-assign]
    env._standing_joint_push_reward_skip_installed = True


DUAL_FINETUNE_LEGS_ACTIVE_ATTR = "_dual_finetune_legs_active"
# Per-env common_step_counter recorded on each legs→wheels handoff (before env.step).
DUAL_FINETUNE_WHEELS_HANDOFF_STEP_ATTR = "_dual_finetune_wheels_handoff_step"


def enable_dual_finetune_legs_reward_skip(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None = None,
):
    """Wrap ``reward_manager.compute`` to zero rewards while the frozen legs policy is active.

    The dual-finetune runner sets ``env._dual_finetune_legs_active`` (bool tensor, one per
    env) before each ``step``. Skipped steps are subtracted from episode sums so
    ``Episode_Reward/*`` logs match wheels-only training signal.
    """
    del env_ids
    if getattr(env, "_dual_finetune_legs_reward_skip_installed", False):
        return

    if not hasattr(env, DUAL_FINETUNE_LEGS_ACTIVE_ATTR):
        setattr(
            env,
            DUAL_FINETUNE_LEGS_ACTIVE_ATTR,
            torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device),
        )

    reward_manager = env.reward_manager
    original_compute = reward_manager.compute

    def compute_with_legs_skip(dt: float) -> torch.Tensor:
        reward_buf = original_compute(dt)
        mask = getattr(env, DUAL_FINETUNE_LEGS_ACTIVE_ATTR, None)
        if mask is None or not torch.any(mask):
            return reward_buf
        mask = mask.to(device=reward_buf.device)
        for term_idx, name in enumerate(reward_manager._term_names):
            term_value = reward_manager._step_reward[:, term_idx] * dt
            reward_manager._episode_sums[name][mask] -= term_value[mask]
            reward_manager._step_reward[mask, term_idx] = 0.0
        reward_buf[mask] = 0.0
        return reward_buf

    reward_manager.compute = compute_with_legs_skip  # type: ignore[method-assign]
    env._dual_finetune_legs_reward_skip_installed = True


def _ensure_dual_finetune_wheels_handoff_step_buf(env: ManagerBasedEnv) -> torch.Tensor:
    """Allocate (if needed) and return the legs→wheels handoff step buffer."""
    num_envs = env.scene.num_envs
    buf = getattr(env, DUAL_FINETUNE_WHEELS_HANDOFF_STEP_ATTR, None)
    if buf is None or buf.shape[0] != num_envs:
        buf = torch.full((num_envs,), -1_000_000_000, device=env.device, dtype=torch.long)
        setattr(env, DUAL_FINETUNE_WHEELS_HANDOFF_STEP_ATTR, buf)
    return buf


def record_dual_finetune_wheels_handoff(env: ManagerBasedEnv, handoff_mask: torch.Tensor) -> None:
    """Record legs→wheels handoffs at the current ``common_step_counter`` (call before ``step``).

    ``ManagerBasedRLEnv.step`` increments ``common_step_counter`` before rewards, so the
    first wheels reward after a handoff sees ``steps_since == 1``.
    """
    if not torch.any(handoff_mask):
        return
    buf = _ensure_dual_finetune_wheels_handoff_step_buf(env)
    mask = handoff_mask.to(device=buf.device)
    buf[mask] = int(env.common_step_counter)


def dual_finetune_action_rate_handoff_boost_mask(
    env: ManagerBasedEnv, duration_s: float
) -> torch.Tensor:
    """True for envs still inside the post-handoff reward boost window."""
    buf = getattr(env, DUAL_FINETUNE_WHEELS_HANDOFF_STEP_ATTR, None)
    if buf is None:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    steps_since = env.common_step_counter - buf
    time_since = steps_since.to(dtype=torch.float32) * float(env.step_dt)
    return (steps_since > 0) & (time_since <= duration_s)


def enable_dual_finetune_action_rate_handoff_boost(
    env: ManagerBasedEnv,
    duration_s: float = 0.5,
    scale: float = 2.0,
    term_name: str = "action_rate_l2",
    term_names: Sequence[str] | None = None,
):
    """Wrap ``reward_manager.compute`` to scale reward terms after legs→wheels handoffs.

    Install after :func:`enable_dual_finetune_legs_reward_skip` so legs-active zeros still
    apply first. The dual-finetune runner records handoffs via
    :func:`record_dual_finetune_wheels_handoff` before each ``step``.

    ``term_names`` selects which reward terms to scale (default: ``[term_name]``).
    Missing terms are skipped silently.
    """
    if getattr(env, "_dual_finetune_action_rate_handoff_boost_installed", False):
        return
    duration_s = float(duration_s)
    scale = float(scale)
    if duration_s <= 0.0 or scale == 1.0:
        return

    names = list(term_names) if term_names is not None else [term_name]
    if not names:
        return

    _ensure_dual_finetune_wheels_handoff_step_buf(env)

    reward_manager = env.reward_manager
    term_indices = [
        (name, reward_manager._term_names.index(name))
        for name in names
        if name in reward_manager._term_names
    ]
    if not term_indices:
        return
    original_compute = reward_manager.compute
    extra_scale = scale - 1.0

    def compute_with_handoff_boost(dt: float) -> torch.Tensor:
        reward_buf = original_compute(dt)
        mask = dual_finetune_action_rate_handoff_boost_mask(env, duration_s)
        if not torch.any(mask):
            return reward_buf
        mask = mask.to(device=reward_buf.device)
        # ``_step_reward`` stores weight * raw (rate); episode sums use rate * dt.
        for name, term_idx in term_indices:
            extra = extra_scale * reward_manager._step_reward[:, term_idx] * dt
            reward_manager._episode_sums[name][mask] += extra[mask]
            reward_manager._step_reward[mask, term_idx] *= scale
            reward_buf[mask] += extra[mask]
        return reward_buf

    reward_manager.compute = compute_with_handoff_boost  # type: ignore[method-assign]
    env._dual_finetune_action_rate_handoff_boost_installed = True


def push_standing_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
):
    """Add random joint-position offsets to standing (zero-command) environments.

    Only environments flagged as standing by the velocity command term are affected.
    Offsets are applied to the current joint positions and clamped to soft joint limits.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    elif env_ids.numel() == 0:
        return
    else:
        env_ids = env_ids.to(device=env.device)

    command_term = env.command_manager.get_term(command_name)
    standing_mask = command_term.is_standing_env[env_ids]
    env_ids = env_ids[standing_mask]
    if env_ids.numel() == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    joint_pos = asset.data.joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    asset.write_joint_position_to_sim(joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    record_standing_joint_push(env, env_ids)


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Note: If "pits" terrain exists, environments on pit terrain will be reset to default state without random
    perturbations to avoid the robot falling into the pit.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Separate pit and non-pit environments
    # Check which environments are assigned to pit terrain (not random reset)
    assigned_to_pits = is_env_assigned_to_terrain(env, "pits")
    pit_env_ids = env_ids[assigned_to_pits[env_ids]]
    non_pit_env_ids = env_ids[~assigned_to_pits[env_ids]]

    # Reset pit environments to default state (no random perturbations)
    if len(pit_env_ids) > 0:
        root_states = asset.data.default_root_state[pit_env_ids].clone()
        positions = root_states[:, 0:3] + env.scene.env_origins[pit_env_ids]
        orientations = root_states[:, 3:7]
        velocities = torch.zeros_like(root_states[:, 7:13])
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=pit_env_ids)

    # Reset non-pit environments with random perturbations
    if len(non_pit_env_ids) > 0:
        root_states = asset.data.default_root_state[non_pit_env_ids].clone()

        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        positions = root_states[:, 0:3] + env.scene.env_origins[non_pit_env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=non_pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=non_pit_env_ids)


_WTW_FREQ_BIN_REWARD_TERM_NAMES = (
    "wtw_tracking_contacts_shaped_force",
    "wtw_tracking_contacts_shaped_vel",
    "wtw_feet_clearance_cmd_linear",
    "wtw_raibert_heuristic",
    "wtw_orientation_control",
    "wtw_jump",
)
_WTW_FREQ_BINS = (
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
    (2.5, 3.0),
)


def enable_wtw_frequency_bin_reward_logging(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "gait_behavior",
    term_names: tuple[str, ...] = _WTW_FREQ_BIN_REWARD_TERM_NAMES,
    freq_bins: tuple[tuple[float, float], ...] = _WTW_FREQ_BINS,
):
    """Log per-bin episode means of WTW rewards vs ``frequency_base`` (does not change training).

    Intended to be registered as a ``startup`` event. Each step assigns the env to a
    ``frequency_base`` bin and accumulates that step's weighted WTW rewards. On reset,
    logs the **conditional** mean ``E[reward | freq in bin]`` as ``WTW_Freq/{term}/{lo}_{hi}``
    (only envs that spent time in the bin). Also logs ``WTW_Freq/bin_frac/{lo}_{hi}`` so
    empty/rare bins are visible. The last bin is inclusive of its upper edge.
    """
    del env_ids
    if getattr(env, "_wtw_frequency_bin_reward_logging_installed", False):
        return

    reward_manager = env.reward_manager
    original_compute = reward_manager.compute
    original_reset = reward_manager.reset
    name_to_idx = {name: idx for idx, name in enumerate(reward_manager._term_names)}
    tracked = [(name, name_to_idx[name]) for name in term_names if name in name_to_idx]
    if not tracked:
        env._wtw_frequency_bin_reward_logging_installed = True
        return

    tracked_names = [name for name, _ in tracked]
    tracked_idx = torch.tensor([idx for _, idx in tracked], device=env.device, dtype=torch.long)
    bin_lo = torch.tensor([lo for lo, _ in freq_bins], device=env.device)
    bin_hi = torch.tensor([hi for _, hi in freq_bins], device=env.device)
    bin_labels = [f"{lo:g}_{hi:g}" for lo, hi in freq_bins]
    num_bins = len(freq_bins)
    num_terms = len(tracked)
    # (B, T, N) reward*dt sums and (B, N) time-in-bin.
    reward_sum = torch.zeros(num_bins, num_terms, env.num_envs, device=env.device)
    time_sum = torch.zeros(num_bins, env.num_envs, device=env.device)
    nan = torch.tensor(float("nan"), device=env.device)

    def compute_with_freq_bin_log(dt: float) -> torch.Tensor:
        reward_buf = original_compute(dt)
        freq = env.command_manager.get_term(command_name).frequency_base.unsqueeze(1)
        in_bin = (freq >= bin_lo) & (freq < bin_hi)
        in_bin[:, -1] = (freq.squeeze(1) >= bin_lo[-1]) & (freq.squeeze(1) <= bin_hi[-1])
        in_bin_bt = in_bin.transpose(0, 1).float()  # (B, N)
        time_sum.add_(in_bin_bt * dt)
        step_rewards = reward_manager._step_reward[:, tracked_idx].T  # (T, N)
        reward_sum.add_(in_bin_bt.unsqueeze(1) * step_rewards.unsqueeze(0) * dt)
        return reward_buf

    def reset_with_freq_bin_log(reset_env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        extras = original_reset(reset_env_ids)
        ids = slice(None) if reset_env_ids is None else reset_env_ids
        episode_time = (env.episode_length_buf[ids].float() * env.step_dt).clamp(min=1.0e-8)
        for bin_idx, label in enumerate(bin_labels):
            bin_time = time_sum[bin_idx, ids]
            extras[f"WTW_Freq/bin_frac/{label}"] = torch.mean(bin_time / episode_time)
            visited = bin_time > 0.0
            for term_idx, name in enumerate(tracked_names):
                key = f"WTW_Freq/{name}/{label}"
                if bool(visited.any()):
                    extras[key] = torch.mean(reward_sum[bin_idx, term_idx, ids][visited] / bin_time[visited])
                else:
                    extras[key] = nan.clone()
        time_sum[:, ids] = 0.0
        reward_sum[:, :, ids] = 0.0
        return extras

    reward_manager.compute = compute_with_freq_bin_log  # type: ignore[method-assign]
    reward_manager.reset = reset_with_freq_bin_log  # type: ignore[method-assign]
    env._wtw_frequency_bin_reward_logging_installed = True
