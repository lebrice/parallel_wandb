"""An example of how to do logging to multiple wandb runs in parallel."""
from .log import wandb_init, wandb_log, map_fn_and_log_to_wandb
__all__ = ["wandb_init", "wandb_log", "map_fn_and_log_to_wandb"]
