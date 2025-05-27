"""Functions that make it easy to create and log metrics to multiple wandb runs in parallel."""

import functools
import operator
from logging import getLogger
from typing import Any

import numpy as np
import optree
from wandb.sdk.wandb_run import Run

from parallel_wandb.utils import NestedSequence, get_step, is_tracer, shape_begins_with, slice

logger = getLogger(__name__)


def wandb_log(
    wandb_run: Run | NestedSequence[Run],
    metrics: dict[str, Any],
    step: int | np.typing.NDArray[np.integer] | np.typing.ArrayLike,
    run_index: np.typing.NDArray[np.integer] | np.typing.ArrayLike | None = None,
    same_metrics_for_all_runs: bool | None = None,
):
    """Log metrics to wandb using `wandb.log` for each run in `wandb_run`.

    If `log_metric_to_all_runs` is False, the metrics are logged to every run.
    """

    wandb_run_array = np.asanyarray(wandb_run)
    multiple_runs = wandb_run_array.size > 1

    if multiple_runs and same_metrics_for_all_runs is None:
        same_metrics_for_all_runs = _check_shape_prefix(metrics, wandb_run_array.shape)  # type: ignore

    # TODO: Probably won't work correctly if only one of `step` or `metrics` is traced.
    this_is_being_traced = optree.tree_any(optree.tree_map(is_tracer, (metrics, step)))  # type: ignore
    this_is_being_vmapped = (
        this_is_being_traced and multiple_runs and same_metrics_for_all_runs is False
    )
    if this_is_being_traced:
        logger.debug(
            f"Logging to wandb under a tracing context: {wandb_run_array.shape=}, "
            f"{same_metrics_for_all_runs=}, {this_is_being_vmapped=}"
        )

    if this_is_being_vmapped:
        # Multiple wandb_runs and metrics are for a single run, this is probably being called
        # from a function that is (or is going to be?) vmapped.
        logger.debug(
            f"Assuming that the calling function is vmapped since {wandb_run_array.ndim=} and {same_metrics_for_all_runs=}"
        )
        # There are multiple wandb runs, and metrics are not stacked
        # (dont have the wandb_runs shape as a prefix in their shapes)
        # --> This is probably being called inside a function that is being vmapped!
        if run_index is None:
            raise ValueError(
                f"There are multiple wandb runs, some metrics are tracers, and metrics are not stacked "
                f"(they dont have the {wandb_run_array.shape=} as a prefix in their shapes). \n"
                f"This indicates that you are likely calling `{wandb_log.__name__}` inside a function "
                f"that is being vmapped, which is great!\n"
                f"However, since we can't tell at which 'index' in the vmap we're at, "
                f"you need to pass the `run_index` argument. "
                f"This array will be used to index into `wandb_runs` to select which run to log at.\n"
                f"`run_index=jnp.arange(num_seeds)` is a good option.\n"
                f"See the `jax_mnist.py` example in the GitHub repo for an example.\n"
                f"Metric shapes: {optree.tree_map(operator.attrgetter('shape'), metrics)}"  # type: ignore
            )

        assert not isinstance(wandb_run, Run)
        return wandb_log_under_vmap(wandb_run, metrics=metrics, step=step, run_index=run_index)

    def log(wandb_run: Run, metrics: dict[str, Any], step: int | np.typing.ArrayLike):
        """Base case: single run, simple dict of metrics."""
        if this_is_being_traced:
            import jax.experimental  # type: ignore
            # IDEA: Try using the sharding argument to io_callback to only log from the first device?

            # TODO: Wandb docs say: "The step must always increase, and it is not
            # possible to log to a previous step." (https://docs.wandb.ai/ref/python/log/#the-wb-step)
            # This implies that our approach with io_callback(ordered=False) is wrong!
            # However, it does seem to work just fine in practice.. 🤔
            if not is_tracer(step):
                step = int(step.item())
                return jax.experimental.io_callback(
                    lambda _metrics: wandb_run.log(_metrics, step=step), (), metrics
                )
            if not optree.tree_all(optree.tree_map(is_tracer, metrics)):
                return jax.experimental.io_callback(
                    lambda _step: wandb_run.log(metrics, step=_step), (), step
                )
            # Everything is a tracer.
            # TODO: actually, part of the metrics could be tracers, and part not.
            return jax.experimental.io_callback(wandb_run.log, (), metrics, step)

        if isinstance(step, np.ndarray) or (
            hasattr(step, "ndim") and callable(getattr(step, "item", None))
        ):
            assert step.ndim == 0, step  # type: ignore
            step = step.item()  # type: ignore
        assert isinstance(step, int), step
        return wandb_run.log(metrics, step=step)

    if wandb_run_array.size == 1:
        wandb_run = wandb_run if isinstance(wandb_run, Run) else wandb_run_array.item()
        assert isinstance(wandb_run, Run)
        return log(wandb_run=wandb_run, metrics=metrics, step=step)

    # non-recursive version that indexes using the multi-dimensional metrics.
    _num_runs = np.prod(wandb_run_array.shape)
    for run_index, wandb_run_i, metrics_i in slice(
        wandb_run_array.shape,
        wandb_run_array,
        metrics,
        # todo: Make this an argument for a more precise and predictable behaviour?
        # strict=False,
        strict=(not same_metrics_for_all_runs) if same_metrics_for_all_runs is not None else True,
    ):
        assert isinstance(wandb_run_i, Run)
        indexing_tuple = np.unravel_index(run_index, wandb_run_array.shape)
        # if not metrics_are_stacked:
        #     metrics_i = metrics
        # logger.info("Run index: %s, metrics: %s", run_index, jax.tree.map(jax.typeof, metrics))
        step_i = get_step(step, indexing_tuple)
        log(wandb_run_i, metrics=metrics_i, step=step_i)

    return


def wandb_log_under_vmap(
    wandb_run: NestedSequence[Run],
    run_index: np.typing.NDArray[np.integer] | np.typing.ArrayLike,
    metrics: dict[str, Any],
    step: np.typing.NDArray[np.integer] | np.typing.ArrayLike,
):
    """WIP: Call to wandb.log inside a function that is vmapped, such as a `train_step`-esque function.

    In this scenario:
    - This function is being vmapped to train multiple runs in parallel.
    - wandb_run is an array of wandb runs
    - `metrics` is a dictionary of metrics to log, but it is NOT stacked!
        - We're only seeing things from the perspective of a single run! (TODO: Unclear why exactly)
    - We don't know which "run index" we're in --> `run_index` needs to be passed in.
    """
    import jax
    import jax.experimental

    # jax.debug.print("Vmapped Logging at step {} {} for run {}.", step, metrics, run_index)
    wandb_run_array = np.asanyarray(wandb_run)

    def log(metrics: dict[str, Any], step: int, run_index: int | tuple[int, ...]):
        if not isinstance(step, int):
            step = step.item()
        run = wandb_run_array[run_index]
        assert isinstance(run, Run)
        run.log(metrics, step=step)

    # The metrics should not be stacked!
    # We're inside vmap, so we should only have the metrics for a single run
    # (Not 100% clear why though).
    assert not _check_shape_prefix(metrics, wandb_run_array.shape)

    jax.experimental.io_callback(
        log,
        (),
        metrics,
        step=step,
        run_index=run_index,
        # TODO: look at the sharding argument to io_callback to only log from the first device?
        # Seems incompatible with vmap though for some reason?
    )


def _merge[T](v1: T, v2: T) -> T:
    """Merge two values (maybe dictionaries) recursively."""
    if not isinstance(v1, dict):
        return v2
    assert isinstance(v2, dict)  # both should be dicts!
    # T := dict
    result = {}
    for k in v1.keys() | v2.keys():
        if k not in v1:
            result[k] = v2[k]
        elif k not in v2:
            result[k] = v1[k]
        else:
            result[k] = _merge(v1[k], v2[k])
    return result  # type: ignore


def _check_shape_prefix(metrics: dict[str, Any], shape: tuple[int, ...]) -> bool:
    """Returns `True` if all the entries in `metrics` have a shape that begins with `shape`."""
    fn = functools.partial(shape_begins_with, prefix=shape)
    return optree.tree_all(optree.tree_map(fn, metrics))
