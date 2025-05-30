import dataclasses
import functools
import logging
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec

import numpy as np
import optree
from wandb.sdk.wandb_run import Run

from parallel_wandb.log import _check_shape_prefix
from parallel_wandb.utils import NestedSequence, get_step, is_tracer, slice

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LogContext:
    run: Run
    run_index: int
    num_runs: int
    step: int


P = ParamSpec("P")


def map_fn_foreach_run(
    wandb_run: Run | NestedSequence[Run],
    fn: Callable[Concatenate[LogContext, P], dict[str, Any] | None],
    step: int | np.typing.ArrayLike,
    run_index: np.typing.NDArray[np.integer] | np.typing.ArrayLike | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
):
    """Map a function over the sliced arg and kwargs for each run and log the results to wandb.

    This is meant to be used to log things like wandb tables, images and such, that
    need to be created with the data of each run.

    In the case of Jax, this function will executed inside a `jax.experimental.io_callback`.

    `fn` should be a function that takes a `LogContext` as first argument.
    This is dataclass that contains the run, run index, number of runs, and current step.
    The function should can either return a dictionary of things to log to wandb or
    do the logging directly using the context's `run` attribute.

    - If `wandb_run` is a single run, the function will be called with the args
      and kwargs unchanged.
    - If `wandb_run` is an array of runs, the function will be called with the
      sliced args and kwargs for each run.

    Arguments:
        wandb_run: Wandb run or ndarray of Wandb runs. Can have more than one dimension.
        fn: Function to call with the context and sliced args and kwargs.
        step: The current step. Can either be an int or an array of the same shape as the runs array.
        run_index: Array of the same shape as `wandb_run` that needs to be passed if this is to be vmapped.
        args: The positional arguments to the function.
        kwargs: The keyword arguments to the function.

    ## Example

    ```python
    import wandb
    from parallel_wandb import wandb_init, map_fn_foreach_run
    import numpy as np

    seeds = np.arange(5)
    runs = wandb_init(
        {"config": {"seed": seeds}},
        project="test_project",
        group="testing",
    )
    # some data for each run
    images = np.stack(
        [np.random.default_rng(seed).uniform(0, 255, (32, 32, 3)).astype(np.uint8) for seed in seeds]
    )
    # Create a wandb.Image with the data for each run:
    step = 0
    map_fn_foreach_run(
        runs,
        lambda ctx, image: {"train_samples": wandb.Image(image)},
        step=step,
        image=images,
    )
    ```
    """
    wandb_run_array = np.asanyarray(wandb_run)

    if all(run.disabled for run in wandb_run_array.flatten()):
        return
    multiple_runs = wandb_run_array.size > 1
    this_is_being_traced = optree.tree_any(
        optree.tree_map(is_tracer, (step, run_index, args, kwargs))
    )  # type: ignore
    logger.debug(f"Logging to wandb with {wandb_run_array.shape=} and {this_is_being_traced=}")
    metrics_are_stacked = _check_shape_prefix((args, kwargs), wandb_run_array.shape)

    def log(
        wandb_run: Run | np.ndarray,
        step: int | np.typing.ArrayLike,
        run_index: int,
        num_runs: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        """Base case: single run, simple dict of metrics."""
        if not isinstance(wandb_run, Run):
            if wandb_run.size == 1:
                wandb_run = wandb_run.item()
            else:
                indexing_tuple = np.unravel_index(run_index, wandb_run_array.shape)
                wandb_run = np.asarray(wandb_run)[*indexing_tuple]
        assert isinstance(wandb_run, Run), wandb_run

        if not isinstance(step, int):
            step = int(step.item())

        log_context = LogContext(run=wandb_run, run_index=run_index, num_runs=num_runs, step=step)
        metrics = fn(log_context, *args, **kwargs)
        assert isinstance(step, int), step
        assert isinstance(metrics, dict) or metrics is None
        if metrics:
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
                f"(they don't have the {wandb_run_array.shape=} as a prefix in their shapes).\n"
                f"This indicates that you are likely calling `{map_fn_foreach_run.__name__}` inside a function "
                f"that is being vmapped, which is great!\n"
                f"However, since we can't tell at which 'index' in the vmap we're at, "
                f"you need to pass the `run_index` argument. "
                f"This array will be used to index into `wandb_runs` to select which run to log at.\n"
                f"`run_index=jnp.arange(num_runs)` is a good option.\n"
                f"See the `jax_mnist.py` example in the GitHub repo for an example.\n"
                f"Metric shapes: {optree.tree_map(jax.typeof, (args, kwargs))}"  # type: ignore
            )

        # raise NotImplementedError(
        #     "TODO! Equivalent of `log_under_vmap` with an additional function to be called inside the io_callback."
        # )

        num_runs = wandb_run_array.size
        if is_tracer(step):
            jax.experimental.io_callback(
                functools.partial(log, wandb_run, num_runs=num_runs),
                (),
                step,
                run_index,
                *args,
                **kwargs,
            )
        else:
            jax.experimental.io_callback(
                functools.partial(log, wandb_run, step, num_runs=num_runs),
                (),
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
            if is_tracer(step_i):
                jax.experimental.io_callback(
                    functools.partial(log, wandb_run_i, run_index, num_runs),
                    (),
                    step_i,
                    *args_i,
                    **kwargs_i,
                )
            else:
                jax.experimental.io_callback(
                    functools.partial(log, step_i, wandb_run_i, run_index, num_runs),
                    (),
                    *args_i,
                    **kwargs_i,
                )
        else:
            log(wandb_run_i, step_i, run_index, num_runs, *args_i, **kwargs_i)
