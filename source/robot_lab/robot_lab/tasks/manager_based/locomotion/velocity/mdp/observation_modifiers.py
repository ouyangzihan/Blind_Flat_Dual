# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Stateful observation modifiers for sim-to-real robustness."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.modifiers import ModifierBase, ModifierCfg


class DelayedObservation(ModifierBase):
    """Delay observations by a per-env lag sampled at reset.

    Lag is in policy control steps (after decimation). Example: with control dt = 0.02 s,
    ``max_lag=2`` delays up to 40 ms. Output shape is unchanged (latest delayed sample only).
    """

    def __init__(self, cfg: DelayedObservationCfg, data_dim: tuple[int, ...], device: str):
        super().__init__(cfg, data_dim, device)
        if cfg.min_lag < 0 or cfg.max_lag < cfg.min_lag:
            raise ValueError(f"Invalid lag range: min_lag={cfg.min_lag}, max_lag={cfg.max_lag}")
        self._min_lag = int(cfg.min_lag)
        self._max_lag = int(cfg.max_lag)
        self._buffer = DelayBuffer(self._max_lag, batch_size=data_dim[0], device=device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            batch_ids = slice(None)
            num = self._data_dim[0]
            lags = torch.randint(
                self._min_lag,
                self._max_lag + 1,
                (num,),
                dtype=torch.int,
                device=self._device,
            )
        else:
            batch_ids = env_ids
            num = len(env_ids) if not isinstance(env_ids, slice) else self._data_dim[0]
            lags = torch.randint(
                self._min_lag,
                self._max_lag + 1,
                (num,),
                dtype=torch.int,
                device=self._device,
            )
        self._buffer.reset(batch_ids)
        self._buffer.set_time_lag(lags, batch_ids)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._buffer.compute(data)


@configclass
class DelayedObservationCfg(ModifierCfg):
    """Configuration for :class:`DelayedObservation`."""

    func: type[DelayedObservation] = DelayedObservation

    min_lag: int = 0
    """Minimum delay in policy steps (inclusive)."""

    max_lag: int = 1
    """Maximum delay in policy steps (inclusive)."""
