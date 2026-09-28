# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Finetune the wheels policy under play_dual-style switching with a frozen legs policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dual-policy finetune: train wheels, freeze legs.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-Wheels-DualFinetune-v0",
    help="Training task (dual-finetune env).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--wheels_task",
    type=str,
    default="RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-Wheels-v0",
    help="Task used to resolve wheels agent config / default checkpoint root.",
)
parser.add_argument(
    "--legs_task",
    type=str,
    default="RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-Legs-v0",
    help="Task used to resolve legs agent config / default checkpoint root.",
)
parser.add_argument("--checkpoint_wheels", type=str, default=None, help="Checkpoint path for the wheels policy.")
parser.add_argument("--checkpoint_legs", type=str, default=None, help="Checkpoint path for the frozen legs policy.")
parser.add_argument(
    "--reload_legs_interval",
    type=int,
    default=None,
    help=(
        "Reload the newest legs checkpoint every N dual-finetune iterations "
        "(default: env reload_legs_interval, currently 500). Set 0 to disable."
    ),
)
parser.add_argument(
    "--wheels_experiment_name",
    type=str,
    default=None,
    help="Experiment folder for wheels checkpoint lookup (default: unitree_go2w_flat_wheels).",
)
parser.add_argument(
    "--legs_experiment_name",
    type=str,
    default=None,
    help="Experiment folder for legs checkpoint lookup (default: unitree_go2w_flat_legs).",
)
parser.add_argument(
    "--wheels_load_run",
    type=str,
    default=None,
    help="Wheels run folder name or absolute path to that run directory.",
)
parser.add_argument(
    "--legs_load_run",
    type=str,
    default=None,
    help="Legs run folder name or absolute path to that run directory.",
)
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
    "--from_scratch",
    action="store_true",
    default=False,
    help=(
        "Train the wheels policy from scratch (no wheels checkpoint). "
        "Uses flat-wheels learning rate and entropy coefficient. "
        "Overrides env_cfg.train_from_scratch when set."
    ),
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import re
import time
from datetime import datetime

import gymnasium as gym
import torch
from dual_finetune_runner import DualFinetuneOnPolicyRunner, wrap_env_for_legs_checkpoint
from entropy_schedule import install_entropy_coef_schedule
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.flat_legs_env_cfg import (
    _PPO_ENTROPY_COEF,
    _PPO_ENTROPY_COEF_LATE,
    _PPO_ENTROPY_DROP_ITERATION,
)

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _normalize_load_run(
    load_run: str | None, default_log_root: str
) -> tuple[str, str]:
    """Return ``(log_root, run_regex)`` for ``get_checkpoint_path``.

    ``load_run`` may be a run folder name, a path to that folder, or a ``.pt`` checkpoint.
    """
    if not load_run:
        return default_log_root, ".*"
    path = os.path.abspath(os.path.expanduser(load_run))
    if os.path.isfile(path):
        run_dir = os.path.dirname(path)
        return os.path.dirname(run_dir), re.escape(os.path.basename(run_dir))
    if os.path.isdir(path):
        return os.path.dirname(path), re.escape(os.path.basename(path.rstrip(os.sep)))
    return default_log_root, load_run


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
    log_root_path, run_pattern = _normalize_load_run(load_run, log_root_path)
    checkpoint_pattern = load_checkpoint if load_checkpoint is not None else r"model_.*\.pt"
    return get_checkpoint_path(log_root_path, run_pattern, checkpoint_pattern)


def _resolve_newest_legs_checkpoint(
    checkpoint_arg: str | None,
    experiment_name: str | None,
    load_run: str | None,
    default_experiment_name: str,
) -> str:
    """Newest ``model_*.pt`` in the legs run, including while that job is still training.

    If ``checkpoint_arg`` is set, stay in that checkpoint's run folder (do not jump runs).
    Otherwise use ``experiment_name`` / ``load_run`` the same way as the initial resolve.
    """
    if checkpoint_arg:
        ckpt = retrieve_file_path(checkpoint_arg)
        run_dir = os.path.dirname(os.path.abspath(ckpt))
        log_root_path = os.path.dirname(run_dir)
        run_name = os.path.basename(run_dir)
        return get_checkpoint_path(log_root_path, re.escape(run_name), r"model_.*\.pt")
    return _resolve_checkpoint(None, experiment_name, load_run, None, default_experiment_name)


def _file_is_stable(path: str, checks: int = 2, interval_s: float = 0.5) -> bool:
    """Return True if file size is unchanged across a few polls (avoids half-written ckpts)."""
    last_size = None
    for _ in range(checks):
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        if size <= 0:
            return False
        if last_size == size:
            return True
        last_size = size
        time.sleep(interval_s)
    return os.path.isfile(path) and os.path.getsize(path) == last_size


def _make_reload_legs_fn(
    legs_runner: OnPolicyRunner,
    *,
    checkpoint_arg: str | None,
    experiment_name: str | None,
    load_run: str | None,
    default_experiment_name: str,
    device: str,
    last_path: list[str],
):
    load_cfg = {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
    }

    def _reload():
        try:
            path = _resolve_newest_legs_checkpoint(
                checkpoint_arg, experiment_name, load_run, default_experiment_name
            )
        except ValueError as exc:
            print(f"[WARN]: Could not resolve a newer legs checkpoint ({exc}).")
            return None
        if path == last_path[0]:
            return None
        if not _file_is_stable(path):
            print(f"[WARN]: Legs checkpoint still being written, skip reload: {path}")
            return None
        try:
            legs_runner.load(path, load_cfg=load_cfg, map_location=device)
        except Exception as exc:  # noqa: BLE001 — concurrent save can leave a partial file
            print(f"[WARN]: Failed to reload legs checkpoint {path}: {exc}")
            return None
        policy = legs_runner.get_inference_policy(device=device)
        for param in policy.parameters():
            param.requires_grad_(False)
        policy.eval()
        last_path[0] = path
        print(f"[INFO]: Reloaded legs policy from: {path}")
        return policy

    return _reload


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Dual-finetune: update wheels only under play_dual switching."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Always log under the wheels experiment directory.
    agent_cfg.experiment_name = args_cli.wheels_experiment_name or agent_cfg.experiment_name or "unitree_go2w_flat_wheels"
    if not agent_cfg.run_name:
        agent_cfg.run_name = "dual_finetune"

    # From-scratch: random wheels init + flat-wheels LR / entropy (not finetune defaults).
    train_from_scratch = args_cli.from_scratch or bool(getattr(env_cfg, "train_from_scratch", False))
    env_cfg.train_from_scratch = train_from_scratch
    if train_from_scratch and agent_cfg.run_name == "dual_finetune":
        agent_cfg.run_name = "dual_from_scratch"

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported when using CPU device.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager based RL environments.")

    env_cfg.log_dir = log_dir

    wheels_task = args_cli.wheels_task.split(":")[-1]
    legs_task = args_cli.legs_task.split(":")[-1]
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

    if train_from_scratch:
        # Match UnitreeGo2WFlatWheelsPPORunnerCfg learning rate. Entropy follows the
        # flat-legs schedule: _PPO_ENTROPY_COEF until _PPO_ENTROPY_DROP_ITERATION, then late.
        agent_cfg.algorithm.learning_rate = 2e-3  # wheels_agent_cfg.algorithm.learning_rate
        agent_cfg.algorithm.entropy_coef = _PPO_ENTROPY_COEF
        agent_cfg.entropy_coef_late = _PPO_ENTROPY_COEF_LATE
        agent_cfg.entropy_drop_iteration = _PPO_ENTROPY_DROP_ITERATION
        print(
            "[INFO]: Training wheels from scratch "
            f"(learning_rate={agent_cfg.algorithm.learning_rate}, "
            f"entropy_coef={agent_cfg.algorithm.entropy_coef} for iterations < "
            f"{agent_cfg.entropy_drop_iteration}, then {agent_cfg.entropy_coef_late})."
        )

    wheels_checkpoint = None
    if not train_from_scratch:
        wheels_checkpoint = _resolve_checkpoint(
            args_cli.checkpoint_wheels or (args_cli.checkpoint if args_cli.resume else None),
            args_cli.wheels_experiment_name or wheels_agent_cfg.experiment_name,
            args_cli.wheels_load_run or args_cli.load_run,
            None,
            wheels_agent_cfg.experiment_name,
        )
        print(f"[INFO]: Loading wheels policy from: {wheels_checkpoint}")
    legs_checkpoint = _resolve_checkpoint(
        args_cli.checkpoint_legs,
        args_cli.legs_experiment_name or legs_agent_cfg.experiment_name,
        args_cli.legs_load_run,
        None,
        legs_agent_cfg.experiment_name,
    )
    print(f"[INFO]: Loading legs policy from: {legs_checkpoint}")

    reload_legs_interval = (
        args_cli.reload_legs_interval
        if args_cli.reload_legs_interval is not None
        else int(getattr(env_cfg, "reload_legs_interval", 0) or 0)
    )
    env_cfg.reload_legs_interval = reload_legs_interval
    if reload_legs_interval > 0:
        print(
            f"[INFO]: Will reload newest legs checkpoint every {reload_legs_interval} iterations "
            f"(experiment={args_cli.legs_experiment_name or legs_agent_cfg.experiment_name}, "
            f"load_run={args_cli.legs_load_run or '.*'})."
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Frozen legs policy (throwaway runner). Widen obs when the checkpoint includes WTW gait terms.
    legs_env = wrap_env_for_legs_checkpoint(env, legs_checkpoint)
    legs_runner = OnPolicyRunner(legs_env, legs_agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    legs_runner.load(
        legs_checkpoint,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
    )
    legs_policy = legs_runner.get_inference_policy(device=env.unwrapped.device)
    for param in legs_policy.parameters():
        param.requires_grad_(False)
    legs_policy.eval()

    legs_ckpt_holder = [legs_checkpoint]
    reload_legs_fn = None
    if reload_legs_interval > 0:
        reload_legs_fn = _make_reload_legs_fn(
            legs_runner,
            checkpoint_arg=args_cli.checkpoint_legs,
            experiment_name=args_cli.legs_experiment_name or legs_agent_cfg.experiment_name,
            load_run=args_cli.legs_load_run,
            default_experiment_name=legs_agent_cfg.experiment_name,
            device=agent_cfg.device,
            last_path=legs_ckpt_holder,
        )

    runner = DualFinetuneOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
        legs_policy=legs_policy,
        switch_vy_threshold=args_cli.switch_vy_threshold,
        switch_wz_threshold=args_cli.switch_wz_threshold,
        switch_hysteresis=args_cli.switch_hysteresis,
        reload_legs_interval=reload_legs_interval,
        reload_legs_fn=reload_legs_fn,
    )
    runner.add_git_repo_to_log(__file__)
    if wheels_checkpoint is not None:
        runner.load(wheels_checkpoint)
    # Start a fresh finetune counter in the new log directory.
    runner.current_learning_iteration = 0
    if train_from_scratch:
        install_entropy_coef_schedule(
            runner,
            early=float(agent_cfg.algorithm.entropy_coef),
            late=float(agent_cfg.entropy_coef_late),
            drop_iteration=int(agent_cfg.entropy_drop_iteration),
        )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "dual_finetune.yaml"),
        {
            "train_from_scratch": train_from_scratch,
            "wheels_checkpoint": wheels_checkpoint,
            "legs_checkpoint": legs_checkpoint,
            "reload_legs_interval": reload_legs_interval,
            "switch_vy_threshold": args_cli.switch_vy_threshold,
            "switch_wz_threshold": args_cli.switch_wz_threshold,
            "switch_hysteresis": args_cli.switch_hysteresis,
        },
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
