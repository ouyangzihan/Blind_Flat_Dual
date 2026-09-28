# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.flat_legs_env_cfg import (
    _PPO_ENTROPY_COEF,
    _PPO_ENTROPY_COEF_LATE,
    _PPO_ENTROPY_DROP_ITERATION,
    _PPO_LEARNING_RATE,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.symmetry import go2w


@configclass
class UnitreeGo2WRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2w_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2WFlatPPORunnerCfg(UnitreeGo2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_go2w_flat"


def _go2w_symmetry_algorithm_cfg(base: RslRlPpoAlgorithmCfg, *, use_data_augmentation: bool) -> RslRlPpoAlgorithmCfg:
    """Return a PPO algorithm cfg with Go2W left-right symmetry augmentation."""
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=base.value_loss_coef,
        use_clipped_value_loss=base.use_clipped_value_loss,
        clip_param=base.clip_param,
        entropy_coef=base.entropy_coef,
        num_learning_epochs=base.num_learning_epochs,
        num_mini_batches=base.num_mini_batches,
        learning_rate=base.learning_rate,
        schedule=base.schedule,
        gamma=base.gamma,
        lam=base.lam,
        desired_kl=base.desired_kl,
        max_grad_norm=base.max_grad_norm,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=use_data_augmentation,
            data_augmentation_func=go2w.compute_symmetric_states,
        ),
    )


@configclass
class UnitreeGo2WFlatWheelsPPORunnerCfg(UnitreeGo2WFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2w_flat_wheels"
        self.algorithm.learning_rate = 2.0e-3
        self.algorithm = _go2w_symmetry_algorithm_cfg(self.algorithm, use_data_augmentation=False)


@configclass
class UnitreeGo2WFlatLegsPPORunnerCfg(UnitreeGo2WFlatPPORunnerCfg):
    """Flat-legs PPO. Entropy stays at ``entropy_coef`` until ``entropy_drop_iteration``."""

    entropy_coef_late: float | None = None
    """Entropy coefficient used once ``entropy_drop_iteration`` is reached. None keeps ``entropy_coef``."""

    entropy_drop_iteration: int | None = None
    """Learning iteration at which ``entropy_coef`` switches to ``entropy_coef_late``."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2w_flat_legs"
        self.algorithm.learning_rate = _PPO_LEARNING_RATE
        self.algorithm.entropy_coef = _PPO_ENTROPY_COEF
        # train.py drops alg.entropy_coef to the late value once this iteration is reached.
        self.entropy_coef_late = _PPO_ENTROPY_COEF_LATE
        self.entropy_drop_iteration = _PPO_ENTROPY_DROP_ITERATION
        self.algorithm = _go2w_symmetry_algorithm_cfg(self.algorithm, use_data_augmentation=False)


@configclass
class UnitreeGo2WFlatWheelsPPORunnerWithSymmetryCfg(UnitreeGo2WFlatWheelsPPORunnerCfg):
    """Flat-wheels PPO with left-right symmetry data augmentation enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm = _go2w_symmetry_algorithm_cfg(self.algorithm, use_data_augmentation=True)


@configclass
class UnitreeGo2WFlatLegsPPORunnerWithSymmetryCfg(UnitreeGo2WFlatLegsPPORunnerCfg):
    """Flat-legs PPO with left-right symmetry data augmentation enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm = _go2w_symmetry_algorithm_cfg(self.algorithm, use_data_augmentation=True)


@configclass
class UnitreeGo2WFlatWheelsDualFinetunePPORunnerCfg(UnitreeGo2WFlatWheelsPPORunnerCfg):
    """Wheels-only dual finetune; logs under the same flat_wheels experiment folder.

    Default uses a conservative LR / entropy for pretrained-wheels finetuning.
    When ``train_dual_finetune.py --from_scratch`` (or env ``train_from_scratch``) is
    set, the script restores this class's parent flat-wheels learning rate (2e-3)
    and uses entropy 0.005 until iteration 10000, then 0.001.
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2w_flat_wheels"
        self.run_name = "dual_finetune"
        # Conservative LR / entropy so pretrained wheels skill is not washed out.
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.entropy_coef = 0.001


@configclass
class UnitreeGo2WFlatWheelsDualFinetunePPORunnerWithSymmetryCfg(UnitreeGo2WFlatWheelsDualFinetunePPORunnerCfg):
    """Dual-finetune PPO with left-right symmetry data augmentation enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm = _go2w_symmetry_algorithm_cfg(self.algorithm, use_data_augmentation=True)
