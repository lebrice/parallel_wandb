"""Lightweight library to facilitate logging to multiple Weights & Biases runs in parallel."""

from .init import wandb_init
from .map_and_log import map_fn_foreach_run
from .log import wandb_log

__all__ = ["wandb_init", "wandb_log", "map_fn_foreach_run"]
