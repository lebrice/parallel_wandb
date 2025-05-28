import functools
import logging
from collections.abc import Callable
import operator
from typing import Any, Concatenate

import numpy as np
import optree
from wandb.sdk.wandb_run import Run

from parallel_wandb.log import _check_shape_prefix
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
    logger.debug(f"Logging to wandb with {wandb_run_array.shape=} and {this_is_being_traced=}")
    metrics_are_stacked = _check_shape_prefix((args, kwargs), wandb_run_array.shape)

    def log(
        wandb_run: Run,
        step: int | np.typing.ArrayLike,
        run_index: int,
        num_runs: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        """Base case: single run, simple dict of metrics."""
        if not isinstance(wandb_run, Run):
            indexing_tuple = np.unravel_index(run_index, wandb_run_array.shape)
            wandb_run = np.asarray(wandb_run)[indexing_tuple]
        if isinstance(step, np.ndarray) or (
            hasattr(step, "ndim") and callable(getattr(step, "item", None))
        ):
            assert step.ndim == 0, step  # type: ignore
            step = step.item()  # type: ignore
        metrics = fn(run_index, num_runs, *args, **kwargs)
        assert isinstance(step, int), step
        wandb_run.log(metrics, step=step)

    if not multiple_runs and not metrics_are_stacked:
        wandb_run = wandb_run if isinstance(wandb_run, Run) else wandb_run_array.item()
        assert isinstance(wandb_run, Run)
        if this_is_being_traced:
            import jax.experimental  # type: ignore

            assert is_tracer(step), "assuming step is also a tracer for now."
            return jax.experimental.io_callback(
                functools.partial(log, wandb_run, run_index=0, num_runs=1),
                (),
                step,
                *args,
                **kwargs,
            )
        return log(wandb_run, step, 0, 1, *args, **kwargs)

    if multiple_runs and not metrics_are_stacked and this_is_being_traced:
        logger.debug(
            f"This is probably being called from a function that is being vmapped since {multiple_runs=}, {metrics_are_stacked=}"
        )
        import jax  # type: ignore
        import jax.experimental  # type: ignore

        if run_index is None:
            raise ValueError(
                f"There are multiple wandb runs, some metrics are tracers, and metrics are not stacked "
                f"(they dont have the {wandb_run_array.shape=} as a prefix in their shapes). \n"
                f"This indicates that you are likely calling `{map_fn_and_log_to_wandb.__name__}` inside a function "
                f"that is being vmapped, which is great!\n"
                f"However, since we can't tell at which 'index' in the vmap we're at, "
                f"you need to pass the `run_index` argument. "
                f"This array will be used to index into `wandb_runs` to select which run to log at.\n"
                f"`run_index=jnp.arange(num_seeds)` is a good option.\n"
                f"See the `jax_mnist.py` example in the GitHub repo for an example.\n"
                f"Metric shapes: {optree.tree_map(jax.typeof, (args, kwargs))}"  # type: ignore
            )

        # raise NotImplementedError(
        #     "TODO! Equivalent of `log_under_vmap` with an additional function to be called inside the io_callback."
        # )

        num_runs = wandb_run_array.size
        assert is_tracer(step), "assuming step is also a tracer for now."
        jax.experimental.io_callback(
            functools.partial(log, wandb_run, num_runs=num_runs),
            (),
            step,
            run_index,
            *args,
            **kwargs,
        )
        return

    num_runs = wandb_run_array.size
    for run_index, wandb_run_i, args_i, kwargs_i in slice(
        wandb_run_array.shape,
        wandb_run_array,
        args,
        kwargs,
        strict=(not metrics_are_stacked),
    ):
        assert isinstance(wandb_run_i, Run), wandb_run_i
        indexing_tuple = np.unravel_index(run_index, wandb_run_array.shape)
        step_i = get_step(step, indexing_tuple)
        if this_is_being_traced:
            import jax.experimental  # type: ignore

            # logger.debug("args_i=%s, kwargs_i=%s, ", args_i, kwargs_i)
            assert is_tracer(step_i), "assuming step is also a tracer for now."
            jax.experimental.io_callback(
                functools.partial(log, wandb_run_i, run_index, num_runs),
                (),
                step_i,
                *args_i,
                **kwargs_i,
            )
        else:
            log(wandb_run_i, run_index, num_runs, step_i, *args_i, **kwargs_i)
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
