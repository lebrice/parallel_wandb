import logging
from collections.abc import Callable
from typing import Any, Concatenate

import numpy as np
import optree
from wandb.sdk.wandb_run import Run

from parallel_wandb.utils import NestedSequence, get_step, is_tracer, slice

logger = logging.getLogger(__name__)


def map_fn_and_log_to_wandb[**P](
    wandb_run: Run | NestedSequence[Run],
    fn: Callable[Concatenate[int, int, P], dict[str, Any]],
    step: int | np.typing.ArrayLike,
    run_index: np.typing.NDArray[np.integer] | np.typing.ArrayLike | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
):
    """Map a function over the (sliced) arg and kwargs and log the results to wandb.

    This is meant to be used to log things like wandb tables, images and such, that
    need to be created with the data of each run.

    `fn` should be a function that takes a grid position (tuple of ints) in addition
    to args and kwargs, then return a dictionary of stuff to log to wandb.

    - If `wandb_run` is a single run, the function will be called with an empty
      tuple as first argument and the args and kwargs unchanged.
    - If `wandb_run` is a list of runs, the function will be called with the
      current position in the grid as the first argument, followed by the sliced
      args and kwargs.

    This works recursively, so the `wandb_run` can be a list of list of wandb runs, etc.
    """
    wandb_run_array = np.asanyarray(wandb_run)
    multiple_runs = wandb_run_array.size > 1
    this_is_being_traced = optree.tree_any(optree.tree_map(is_tracer, (wandb_run, step)))  # type: ignore

    def log(
        wandb_run: Run,
        step: int | np.typing.ArrayLike,
        run_index: int,
        num_runs: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        """Base case: single run, simple dict of metrics."""
        if isinstance(step, np.ndarray) or (
            hasattr(step, "ndim") and callable(getattr(step, "item", None))
        ):
            assert step.ndim == 0, step  # type: ignore
            step = step.item()  # type: ignore
        metrics = fn(run_index, num_runs, *args, **kwargs)
        assert isinstance(step, int), step
        wandb_run.log(metrics, step=step)

    if not multiple_runs:
        wandb_run = wandb_run if isinstance(wandb_run, Run) else wandb_run_array.item()
        assert isinstance(wandb_run, Run)
        if this_is_being_traced:
            import jax.experimental  # type: ignore

            assert is_tracer(step), "assuming step is also a tracer for now."
            return jax.experimental.io_callback(
                lambda _step, *_args, **_kwargs: log(wandb_run, _step, 0, 1, *_args, **_kwargs),
                (),
                step,
                *args,
                **kwargs,
            )
        return log(wandb_run, step, 0, 1, *args, **kwargs)

    num_runs = wandb_run_array.size
    for run_index, wandb_run_i, args_i, kwargs_i in slice(
        wandb_run_array.shape, wandb_run_array, args, kwargs
    ):
        indexing_tuple = np.unravel_index(run_index, wandb_run_array.shape)
        step_i = get_step(step, indexing_tuple)
        if this_is_being_traced:
            import jax.experimental  # type: ignore

            assert is_tracer(step_i), "assuming step is also a tracer for now."
            jax.experimental.io_callback(
                lambda _step, *_args_i, **_kwargs_i: log(
                    wandb_run_i, _step, run_index, num_runs, *_args_i, **_kwargs_i
                ),
                (),
                step_i,
                *args_i,
                *kwargs_i,
            )
        else:
            log(wandb_run_i, step_i, run_index, num_runs, *args_i, **kwargs_i)
    return
    # Everything is a tracer.
    # TODO: actually, part of the metrics could be tracers, and part not.

    def log_fn(
        wandb_run: Run,
        step: int,
        run_index: int,
        num_runs: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        metrics = fn(run_index, num_runs, *args, **kwargs)
        # Base case: single run, single metric.
        logger.debug("Logging to wandb run %s: metrics=%s step=%s", wandb_run.name, metrics, step)
        wandb_run.log(metrics, step=step)
        return

    for run_index, wandb_run_i, args_i, kwargs_i in slice(
        wandb_run_array.shape, wandb_run_array, args, kwargs
    ):
        indexing_tuple = np.unravel_index(run_index, wandb_run_array.shape)
        step_i = get_step(step, indexing_tuple)
        log_fn(wandb_run_i, step_i, run_index, wandb_run_array.size, *args_i, **kwargs_i)
