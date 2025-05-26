"""An example of how to do logging to multiple wandb runs in parallel."""

from .init import wandb_init
from .map_and_log import map_fn_and_log_to_wandb
from .log import wandb_log

__all__ = ["wandb_init", "wandb_log", "map_fn_and_log_to_wandb"]
