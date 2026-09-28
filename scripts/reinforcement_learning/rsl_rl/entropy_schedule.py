# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Step the PPO entropy coefficient once a learning iteration is reached."""

from __future__ import annotations


def install_entropy_coef_schedule(runner, *, early: float, late: float, drop_iteration: int) -> None:
    """Use ``early`` before ``drop_iteration`` and ``late`` from that iteration on.

    The counter starts at ``runner.current_learning_iteration``, so a resumed run
    keeps the same schedule. PPO reads ``alg.entropy_coef`` inside ``update``.
    """
    state = {"it": int(runner.current_learning_iteration)}
    original_update = runner.alg.update

    def update(*args, **kwargs):
        it = state["it"]
        coef = early if it < drop_iteration else late
        if runner.alg.entropy_coef != coef:
            print(
                f"[INFO]: entropy_coef {runner.alg.entropy_coef} -> {coef} "
                f"at learning iteration {it} (drop at {drop_iteration})."
            )
            runner.alg.entropy_coef = coef
        state["it"] = it + 1
        return original_update(*args, **kwargs)

    runner.alg.update = update
    next_coef = early if state["it"] < drop_iteration else late
    print(
        f"[INFO]: Entropy schedule: {early} for iterations < {drop_iteration}, then {late} "
        f"(next update is iteration {state['it']}, coef={next_coef})."
    )
