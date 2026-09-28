# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""On-policy runner that freezes a legs policy and finetunes wheels on handoffs."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan
from tensordict import TensorDict

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.events import (
    DUAL_FINETUNE_LEGS_ACTIVE_ATTR,
    enable_dual_finetune_action_rate_handoff_boost,
    enable_dual_finetune_legs_reward_skip,
    record_dual_finetune_wheels_handoff,
)

# gait_clock (8) + gait_footswing (1); appended only for legs policies trained with enable_wtw_gait_obs.
_WTW_GAIT_OBS_DIM = 9


def gait_obs_suffix(env) -> torch.Tensor:
    """Return ``[gait_clock (8), gait_footswing (1)]`` from the env gait command."""
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    return torch.cat([mdp.gait_clock(unwrapped), mdp.gait_footswing(unwrapped)], dim=-1)


def append_wtw_gait_obs(obs: TensorDict | torch.Tensor, env) -> TensorDict | torch.Tensor:
    """Append WTW gait terms to policy (and critic) observations."""
    extra = gait_obs_suffix(env)
    if torch.is_tensor(obs):
        return torch.cat([obs, extra.to(device=obs.device, dtype=obs.dtype)], dim=-1)
    out = obs.clone()
    out["policy"] = torch.cat([obs["policy"], extra.to(device=obs["policy"].device, dtype=obs["policy"].dtype)], dim=-1)
    if "critic" in out.keys():
        out["critic"] = torch.cat(
            [obs["critic"], extra.to(device=obs["critic"].device, dtype=obs["critic"].dtype)], dim=-1
        )
    return out


def policy_obs_dim(obs: TensorDict | torch.Tensor) -> int:
    if torch.is_tensor(obs):
        return int(obs.shape[-1])
    return int(obs["policy"].shape[-1])


def policy_in_dim(policy) -> int | None:
    """Actor input width (``MLPModel.obs_dim`` or first Linear)."""
    obs_dim = getattr(policy, "obs_dim", None)
    if obs_dim is not None:
        return int(obs_dim)
    for module in getattr(policy, "modules", lambda: [])():
        if hasattr(module, "in_features"):
            return int(module.in_features)
    return None


def obs_for_legs_policy(obs: TensorDict | torch.Tensor, legs_policy, env) -> TensorDict | torch.Tensor:
    """Widen or slice ``obs`` so it matches the frozen legs actor."""
    expected = policy_in_dim(legs_policy)
    current = policy_obs_dim(obs)
    if expected is None or expected == current:
        return obs
    if expected == current + _WTW_GAIT_OBS_DIM:
        return append_wtw_gait_obs(obs, env)
    if expected < current:
        if torch.is_tensor(obs):
            return obs[..., :expected]
        out = obs.clone()
        out["policy"] = obs["policy"][..., :expected]
        return out
    raise RuntimeError(
        f"Frozen legs policy expects obs dim {expected} but dual-finetune env provides {current}. "
        "A 66-dim legs checkpoint needs gait_behavior on the dual-finetune env."
    )


class LegsGaitObsEnv:
    """Present policy/critic obs with WTW gait terms appended for the frozen legs runner."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def get_observations(self):
        return append_wtw_gait_obs(self._env.get_observations(), self._env)

    def __getattr__(self, name):
        return getattr(self._env, name)


def wrap_env_for_legs_checkpoint(env, checkpoint_path: str):
    """Wrap ``env`` when the legs checkpoint actor is wider by the WTW gait suffix."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    actor_in = None
    if isinstance(state, dict):
        for key, value in state.items():
            if not hasattr(value, "ndim") or value.ndim != 2:
                continue
            if key.endswith("actor.0.weight") or key.endswith("mlp.0.weight"):
                actor_in = int(value.shape[1])
                break
    current = policy_obs_dim(env.get_observations())
    if actor_in == current + _WTW_GAIT_OBS_DIM:
        print(f"[INFO]: Wrapping dual-finetune env for {actor_in}-dim legs actor (appending gait clock/footswing).")
        return LegsGaitObsEnv(env)
    if actor_in is not None and actor_in != current:
        print(
            f"[WARN]: Legs checkpoint actor dim {actor_in} does not match env policy dim {current} "
            f"or {current + _WTW_GAIT_OBS_DIM}; loading as-is."
        )
    return env


def uses_leg_policy(
    vy: torch.Tensor,
    wz: torch.Tensor,
    active_leg_policy: torch.Tensor,
    vy_threshold: float,
    wz_threshold: float,
    hysteresis: float,
) -> torch.Tensor:
    """Per-env play_dual switch rule with hysteresis."""
    enter_legs = (vy.abs() > vy_threshold) | (wz.abs() > wz_threshold)
    stay_legs = (vy.abs() > vy_threshold * hysteresis) | (wz.abs() > wz_threshold * hysteresis)
    return torch.where(active_leg_policy, stay_legs, enter_legs)


class DualFinetuneOnPolicyRunner(OnPolicyRunner):
    """Train the wheels policy while a frozen legs policy drives legs-mode segments.

    Only wheels-active transitions contribute to PPO updates. Legs-active steps still
    step the simulator (to create realistic post-legs states) but their rewards are
    zeroed in the env (including ``Episode_Reward/*`` episode sums) and their advantages
    / value targets are zeroed after GAE so they do not update the wheels networks.
    Wheels→legs boundaries are treated as timeouts so GAE does not cross policies.
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        *,
        legs_policy,
        switch_vy_threshold: float = 0.05,
        switch_wz_threshold: float = 0.05,
        switch_hysteresis: float = 0.5,
        reload_legs_interval: int = 0,
        reload_legs_fn: Callable[[], object] | None = None,
    ) -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.legs_policy = legs_policy
        self.switch_vy_threshold = switch_vy_threshold
        self.switch_wz_threshold = switch_wz_threshold
        self.switch_hysteresis = switch_hysteresis
        self.reload_legs_interval = int(reload_legs_interval)
        self.reload_legs_fn = reload_legs_fn
        self.active_leg_policy = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        unwrapped = self.env.unwrapped
        # Zero env rewards + episode sums while frozen legs policy is active.
        enable_dual_finetune_legs_reward_skip(unwrapped)
        # Scale selected reward terms for a short sim-time window after each legs→wheels handoff.
        cfg = unwrapped.cfg
        enable_dual_finetune_action_rate_handoff_boost(
            unwrapped,
            duration_s=float(getattr(cfg, "action_rate_handoff_boost_duration_s", 0.5)),
            scale=float(getattr(cfg, "action_rate_handoff_boost_scale", 2.0)),
            term_names=getattr(
                cfg,
                "action_rate_handoff_boost_terms",
                ("action_rate_l2",),
            ),
        )

    def _maybe_reload_legs(self, it: int, start_it: int) -> None:
        """Swap in a newer frozen legs policy without interrupting wheels PPO."""
        if self.reload_legs_interval <= 0 or self.reload_legs_fn is None:
            return
        if it <= start_it or it % self.reload_legs_interval != 0:
            return
        updated = self.reload_legs_fn()
        if updated is not None:
            self.legs_policy = updated
        self.legs_policy.eval()

    def _set_legs_reward_skip(self, use_legs: torch.Tensor) -> None:
        """Publish the per-env legs-active mask used by the reward_manager wrapper."""
        unwrapped = self.env.unwrapped
        mask = getattr(unwrapped, DUAL_FINETUNE_LEGS_ACTIVE_ATTR, None)
        if mask is None or mask.shape[0] != use_legs.shape[0]:
            setattr(unwrapped, DUAL_FINETUNE_LEGS_ACTIVE_ATTR, use_legs.to(device=unwrapped.device).clone())
        else:
            mask[:] = use_legs.to(device=mask.device)

    def _command_vy_wz(self) -> tuple[torch.Tensor, torch.Tensor]:
        command = self.env.unwrapped.command_manager.get_command("base_velocity")
        return command[:, 1].to(self.device), command[:, 2].to(self.device)

    def _apply_mode_masks(self, train_mask: torch.Tensor) -> None:
        """Patch rollout storage so GAE/PPO ignore legs-active transitions."""
        st = self.alg.storage
        gamma = self.alg.gamma
        # train_mask: [T, N], True where wheels (or standing) was active.
        wheels = train_mask
        legs = ~wheels
        legs_u = legs.unsqueeze(-1)

        # True terminals for legs steps (no bootstrap through frozen-policy segments).
        st.dones = torch.where(legs_u, torch.ones_like(st.dones), st.dones)
        st.rewards = torch.where(legs_u, torch.zeros_like(st.rewards), st.rewards)

        # Wheels step immediately before a legs step: timeout bootstrap + cut.
        if st.num_transitions_per_env > 1:
            pre_legs = wheels[:-1] & legs[1:]
            pre_legs_u = pre_legs.unsqueeze(-1)
            st.rewards[:-1] = st.rewards[:-1] + gamma * st.values[1:] * pre_legs_u.float()
            st.dones[:-1] = torch.where(pre_legs_u, torch.ones_like(st.dones[:-1]), st.dones[:-1])

    def _zero_legs_advantages(self, train_mask: torch.Tensor) -> None:
        """Remove legs-active samples from the PPO update."""
        st = self.alg.storage
        wheels = train_mask.unsqueeze(-1)
        st.advantages *= wheels.float()
        st.returns = torch.where(wheels, st.returns, st.values)
        if not self.alg.normalize_advantage_per_mini_batch:
            valid = st.advantages[train_mask]
            if valid.numel() > 1:
                st.advantages = (st.advantages - valid.mean()) / (valid.std() + 1e-8)
                st.advantages *= wheels.float()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.legs_policy.eval()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            self._maybe_reload_legs(it, start_it)
            start = time.time()
            train_mask_steps: list[torch.Tensor] = []
            wheels_active_frac = 0.0
            handoff_count = 0.0

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    vy, wz = self._command_vy_wz()
                    use_legs = uses_leg_policy(
                        vy,
                        wz,
                        self.active_leg_policy,
                        self.switch_vy_threshold,
                        self.switch_wz_threshold,
                        self.switch_hysteresis,
                    )
                    handoff_to_wheels = (~use_legs) & self.active_leg_policy
                    self.active_leg_policy = use_legs
                    # Zero reward terms + Episode_Reward sums for legs-active envs this step.
                    self._set_legs_reward_skip(use_legs)
                    # Reset post-handoff action_rate_l2 boost timer (sim time window).
                    record_dual_finetune_wheels_handoff(self.env.unwrapped, handoff_to_wheels)

                    wheels_actions = self.alg.act(obs)
                    legs_actions = self.legs_policy(obs_for_legs_policy(obs, self.legs_policy, self.env))
                    actions = torch.where(use_legs.unsqueeze(-1), legs_actions, wheels_actions)

                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    # Env already zeroes legs rewards; keep an explicit wheels mask for PPO/GAE.
                    wheels_mask = ~use_legs
                    rewards_for_learn = rewards * wheels_mask.float()

                    self.alg.process_env_step(obs, rewards_for_learn, dones, extras)
                    self.logger.process_env_step(rewards_for_learn, dones, extras, None)

                    train_mask_steps.append(wheels_mask)
                    wheels_active_frac += wheels_mask.float().mean().item()
                    handoff_count += handoff_to_wheels.float().sum().item()

                collect_time = time.time() - start
                start = time.time()

                train_mask = torch.stack(train_mask_steps, dim=0)
                self._apply_mode_masks(train_mask)
                self.alg.compute_returns(obs)
                self._zero_legs_advantages(train_mask)

            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it

            steps = max(1, self.cfg["num_steps_per_env"])
            loss_dict = dict(loss_dict)
            loss_dict["wheels_active_frac"] = wheels_active_frac / steps
            loss_dict["handoffs_per_rollout"] = handoff_count

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
