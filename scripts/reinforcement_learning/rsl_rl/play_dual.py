# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play two RSL-RL checkpoints and switch policies at runtime."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play two RSL-RL agents with runtime policy switching.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the simulation environment task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--wheels_task",
    type=str,
    default=None,
    help="Task name for wheels policy agent config. Defaults to --task.",
)
parser.add_argument(
    "--legs_task",
    type=str,
    default=None,
    help="Task name for leg-raising policy agent config. Defaults to --task.",
)
parser.add_argument("--checkpoint_wheels", type=str, default=None, help="Checkpoint path for the wheels policy.")
parser.add_argument("--checkpoint_legs", type=str, default=None, help="Checkpoint path for the leg-raising policy.")
parser.add_argument(
    "--wheels_experiment_name",
    type=str,
    default=None,
    help="Experiment folder name for wheels policy when --checkpoint_wheels is not set.",
)
parser.add_argument(
    "--legs_experiment_name",
    type=str,
    default=None,
    help="Experiment folder name for leg-raising policy when --checkpoint_legs is not set.",
)
parser.add_argument("--wheels_load_run", type=str, default=None, help="Run folder for wheels policy checkpoint lookup.")
parser.add_argument("--legs_load_run", type=str, default=None, help="Run folder for leg-raising policy checkpoint lookup.")
parser.add_argument(
    "--switch_vy_threshold", type=float, default=0.05, help="Enter leg policy when |vy| exceeds this value."
)
parser.add_argument(
    "--switch_wz_threshold", type=float, default=0.05, help="Enter leg policy when |wz| exceeds this value."
)
parser.add_argument(
    "--switch_hysteresis",
    type=float,
    default=0.5,
    help="Multiplier applied to switch thresholds when exiting leg policy.",
)
parser.add_argument(
    "--obs_delay",
    action="store_true",
    default=False,
    help="Keep training observation-delay modifiers. Default: disable delay to match rl_sar deploy.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


def _resolve_checkpoint(
    checkpoint_arg: str | None,
    experiment_name: str | None,
    load_run: str | None,
    load_checkpoint: str | None,
    default_experiment_name: str,
) -> str:
    if checkpoint_arg:
        return retrieve_file_path(checkpoint_arg)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name or default_experiment_name))
    run_pattern = load_run if load_run is not None else ".*"
    checkpoint_pattern = load_checkpoint if load_checkpoint is not None else r"model_.*\.pt"
    return get_checkpoint_path(log_root_path, run_pattern, checkpoint_pattern)


def _create_runner(env, agent_cfg: RslRlBaseRunnerCfg):
    if agent_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "DistillationRunner":
        return DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")


def _checkpoint_mlp_in_dims(path: str) -> tuple[int | None, int | None]:
    """Return (actor_in, critic_in) from an RSL-RL checkpoint first Linear layer."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if not isinstance(state, dict):
        return None, None
    actor_in = None
    critic_in = None
    for key, value in state.items():
        if not hasattr(value, "ndim") or value.ndim != 2:
            continue
        if key.endswith("actor.0.weight"):
            actor_in = int(value.shape[1])
        elif key.endswith("critic.0.weight"):
            critic_in = int(value.shape[1])
    return actor_in, critic_in


class _ObsPrefixEnv:
    """Present a prefix of concatenated obs so a narrower checkpoint can load/run."""

    def __init__(self, env, num_obs: int, num_privileged_obs: int | None = None):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "num_obs", num_obs)
        if num_privileged_obs is not None:
            object.__setattr__(self, "num_privileged_obs", num_privileged_obs)

    def get_observations(self):
        return _adapt_obs_dim(self._env.get_observations(), self.num_obs)

    def __getattr__(self, name):
        return getattr(self._env, name)


def _adapt_obs_dim(obs, in_dim: int | None):
    """Prefix-slice concatenated policy obs when the env is wider than the policy."""
    if in_dim is None or not torch.is_tensor(obs) or obs.shape[-1] == in_dim:
        return obs
    if obs.shape[-1] > in_dim:
        return obs[..., :in_dim]
    raise RuntimeError(
        f"Policy expects obs dim {in_dim} but env provided {obs.shape[-1]}. "
        "Train/play this policy in an env that includes its observation terms."
    )


def _uses_leg_policy(vy: float, wz: float, active_leg_policy: bool) -> bool:
    if active_leg_policy:
        vy_threshold = args_cli.switch_vy_threshold * args_cli.switch_hysteresis
        wz_threshold = args_cli.switch_wz_threshold * args_cli.switch_hysteresis
    else:
        vy_threshold = args_cli.switch_vy_threshold
        wz_threshold = args_cli.switch_wz_threshold
    return abs(vy) > vy_threshold or abs(wz) > wz_threshold


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with two RSL-RL agents and switch between them at runtime."""
    wheels_task = args_cli.wheels_task or args_cli.task.split(":")[-1]
    legs_task = args_cli.legs_task or args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    wheels_agent_cfg = load_cfg_from_registry(wheels_task, args_cli.agent)
    legs_agent_cfg = load_cfg_from_registry(legs_task, args_cli.agent)
    wheels_agent_cfg = handle_deprecated_rsl_rl_cfg(
        cli_args.update_rsl_rl_cfg(wheels_agent_cfg, args_cli), installed_version
    )
    legs_agent_cfg = handle_deprecated_rsl_rl_cfg(
        cli_args.update_rsl_rl_cfg(legs_agent_cfg, args_cli), installed_version
    )

    if args_cli.wheels_experiment_name:
        wheels_agent_cfg.experiment_name = args_cli.wheels_experiment_name
    if args_cli.legs_experiment_name:
        legs_agent_cfg.experiment_name = args_cli.legs_experiment_name

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    env_cfg.observations.policy.enable_corruption = False
    if not args_cli.obs_delay:
        for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
            term = getattr(env_cfg.observations.policy, term_name, None)
            if term is not None:
                term.modifiers = None
        print("[INFO]: Observation delay modifiers disabled (pass --obs_delay to restore training delay).")
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    controller = None
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        if getattr(env_cfg.commands, "gait_behavior", None) is not None:
            env_cfg.commands.gait_behavior.resampling_time_range = (1.0e9, 1.0e9)
        # Wheels env zeros vy/wz command ranges (x-only training); use legs ranges for keyboard.
        legs_env_cfg = load_cfg_from_registry(legs_task, "env_cfg_entry_point")
        legs_ranges = legs_env_cfg.commands.base_velocity.ranges
        vy_sensitivity = max(abs(legs_ranges.lin_vel_y[0]), abs(legs_ranges.lin_vel_y[1]))
        wz_sensitivity = max(abs(legs_ranges.ang_vel_z[0]), abs(legs_ranges.ang_vel_z[1]))
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=vy_sensitivity,
            omega_z_sensitivity=wz_sensitivity,
        )
        print(
            f"[INFO]: Keyboard sensitivities vx={config.v_x_sensitivity}, "
            f"vy={config.v_y_sensitivity}, wz={config.omega_z_sensitivity}"
        )
        controller = Se2Keyboard(config)

    wheels_checkpoint = _resolve_checkpoint(
        args_cli.checkpoint_wheels,
        args_cli.wheels_experiment_name,
        args_cli.wheels_load_run or args_cli.load_run,
        None,
        wheels_agent_cfg.experiment_name,
    )
    legs_checkpoint = _resolve_checkpoint(
        args_cli.checkpoint_legs,
        args_cli.legs_experiment_name,
        args_cli.legs_load_run or args_cli.load_run,
        None,
        legs_agent_cfg.experiment_name,
    )

    log_dir = os.path.dirname(wheels_checkpoint)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_dual"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=wheels_agent_cfg.clip_actions)

    wheels_actor_dim, wheels_critic_dim = _checkpoint_mlp_in_dims(wheels_checkpoint)
    legs_actor_dim, legs_critic_dim = _checkpoint_mlp_in_dims(legs_checkpoint)
    wheels_env = env if wheels_actor_dim is None else _ObsPrefixEnv(env, wheels_actor_dim, wheels_critic_dim)
    legs_env = env if legs_actor_dim is None else _ObsPrefixEnv(env, legs_actor_dim, legs_critic_dim)

    print(f"[INFO]: Loading wheels policy from: {wheels_checkpoint}")
    wheels_runner = _create_runner(wheels_env, wheels_agent_cfg)
    wheels_runner.load(wheels_checkpoint)
    wheels_policy = wheels_runner.get_inference_policy(device=env.unwrapped.device)

    print(f"[INFO]: Loading leg-raising policy from: {legs_checkpoint}")
    legs_runner = _create_runner(legs_env, legs_agent_cfg)
    legs_runner.load(legs_checkpoint)
    legs_policy = legs_runner.get_inference_policy(device=env.unwrapped.device)
    wheels_obs_dim = wheels_actor_dim
    legs_obs_dim = legs_actor_dim

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    active_leg_policy = False
    active_policy_name = "wheels"

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if controller is not None:
                command = controller.advance()
                command_tensor = env.unwrapped.command_manager.get_command("base_velocity")
                command_tensor[:] = torch.tensor(command, dtype=command_tensor.dtype, device=command_tensor.device)
                vy = float(command[1])
                wz = float(command[2])
            else:
                command_tensor = env.unwrapped.command_manager.get_command("base_velocity")
                vy = float(command_tensor[0, 1].item())
                wz = float(command_tensor[0, 2].item())

            use_legs = _uses_leg_policy(vy, wz, active_leg_policy)
            if use_legs != active_leg_policy:
                active_leg_policy = use_legs
                active_policy_name = "legs" if use_legs else "wheels"
                print(f"[INFO]: Switched to {active_policy_name} policy.")

            policy = legs_policy if use_legs else wheels_policy
            policy_obs = _adapt_obs_dim(obs, legs_obs_dim if use_legs else wheels_obs_dim)
            actions = policy(policy_obs)
            obs, _, dones, _ = env.step(actions)

            if version.parse(installed_version) >= version.parse("4.0.0"):
                wheels_policy.reset(dones)
                legs_policy.reset(dones)
            else:
                wheels_runner.alg.policy.reset(dones)
                legs_runner.alg.policy.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
