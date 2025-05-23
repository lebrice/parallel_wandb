"""Functions that make it easy to create and log metrics to multiple wandb runs in parallel."""

import functools
import inspect
import operator
import os
import typing
from collections.abc import Callable, Sequence
from logging import getLogger
from typing import Any, Concatenate, Iterable, Mapping, TypeVar

import numpy as np
import optree
import optree.accessor
import wandb
from optree import PyTree
from wandb.sdk.wandb_run import Run

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
type NestedSequence[T] = Sequence[T | NestedSequence[T]]
type NestedMapping[K, V] = Mapping[K, V | NestedMapping[K, V]]

logger = getLogger(__name__)


# IDEA: only show handles to Jax, put the Run objects in a global variable. (ugly)
# RUN_OBJECTS: dict[int, Run] = {}


def wandb_init[**P](
    stacked_overrides: NestedMapping[str, np.typing.ArrayLike] | None = None,
    process_index: int | None = None,
    _wandb_init: Callable[P, Run] = wandb.init,
    *args: P.args,
    **kwargs: P.kwargs,
) -> NestedSequence[Run]:
    """Initializes multiple wandb runs in parallel.

    The usual args and kwargs to be passed to wandb.init will be overwritten by the (unstacked) values
    in `stacked_overrides`. The values in `stacked_overrides` should be lists or arrays with the same
    shape. The shape of the first item in that dict determines the shape of the runs to be created.
    The stacked arguments are to be passed separately and will override the values from *args and **kwargs.

    For example:

    ```python
    wandb_init({"name": ["run_1", "run_2", "run_3"], "config": {"seed": [1, 2, 3]}})
    # This will create three runs like so:
    np.asarray([
        wandb.init(name="run_1", config={"seed": 1}, reinit="create_new"),
        wandb.init(name="run_2", config={"seed": 2}, reinit="create_new"),
        wandb.init(name="run_3", config={"seed": 3}, reinit="create_new"),
    ])
    ```

    This also works with nested arrays:

    ```python
    wandb_init({"name": [["run_1", "run_2"], ["run_3", "run_4]], "config": {"seed": [[1, 2], [3, 4]]}})
    # This will create four runs like so:
    np.asarray([
        [
            wandb.init(name="run_1", config={"seed": 1}, reinit="create_new"),
            wandb.init(name="run_2", config={"seed": 2}, reinit="create_new"),
        ],
        [
            wandb.init(name="run_3", config={"seed": 3}, reinit="create_new"),
            wandb.init(name="run_4", config={"seed": 4}, reinit="create_new"),
        ]
    ])
    ```

    """
    if optree.tree_any(optree.tree_map(_is_tracer, (stacked_overrides, args, kwargs))):  # type: ignore
        raise ValueError(
            "`wandb_init` is not yet compatible with `jax.jit` or `jax.vmap`.\n"
            "For now, create the runs outside the jitted function, and pass the "
            "runs as a static argument."
        )

    # Disable logging if not on the first process.
    # NOTE: With Jax, it's best to do the same thing on all processes, to avoid deadlocks.
    # For example, we'd create the dicts and things that are to be logged to wandb, and then pass
    # them to disabled runs when process_index != 0.
    # todo: Do we want to enable these goodies by default?
    if process_index is None and (_slurm_proc_id := os.environ.get("SLURM_PROCID")):
        process_index = int(_slurm_proc_id)

    if "SLURM_JOB_ID" in os.environ:
        # Use the job id as the default for the 'group' argument.
        kwargs.setdefault("group", os.environ["SLURM_JOB_ID"])
        config = kwargs.setdefault("config", {})
        assert isinstance(config, dict)
        # Always useful: Add the SLURM environment variables to the config dict.
        config.update({k: v for k, v in os.environ.items() if k.startswith("SLURM")})

    # IDEA: Could be interesting to enable logging on other processes if the data is local to them anyway?
    # (to avoid transferring data to the first node all the time)
    if (process_index or 0) != 0:
        kwargs["mode"] = "disabled"

    def _base_case(*args: P.args, **kwargs: P.kwargs) -> Run:
        kwargs["reinit"] = "create_new"  # Essential: Makes it possible to create multiple runs.
        return _wandb_init(*args, **kwargs)

    if not stacked_overrides:
        return np.asanyarray(_base_case(*args, **kwargs))

    stacked_overrides = stacked_overrides or {}
    _stacked_overrides = typing.cast(Any, stacked_overrides)  # typing bug in optree?
    accessors, overrides, _overrides_treedef = optree.tree_flatten_with_accessor(
        _stacked_overrides,
        is_leaf=lambda v: isinstance(v, (tuple | list | np.ndarray)) or hasattr(v, "shape"),
    )

    first_override = overrides[0]
    if not (isinstance(first_override, Sequence) or hasattr(first_override, "shape")):
        # The overrides are not stacked! (weird!) Do we want to support this?
        raise NotImplementedError(
            f"Assuming that all overrides are stacked for now. {first_override=}, {stacked_overrides=}"
        )

    overrides = list(map(np.asarray, overrides))

    shape = overrides[0].shape  # assumed shared across all overrides.
    n_runs = int(np.prod(shape))

    sig = inspect.signature(wandb.init)
    base_bound_args = sig.bind_partial(*args, **kwargs)
    runs = []
    for run_index in range(n_runs):
        # Unravel the index to get the position in the grid.
        grid_pos = np.unravel_index(run_index, shape)
        # Get the overrides for this run.

        _overrides = typing.cast(Any, overrides)  # typing bug in optree (list isn't a pytree?)
        overrides_i = optree.tree_map(operator.itemgetter(grid_pos), _overrides)

        override_bound_args = sig.bind_partial(*base_bound_args.args, **base_bound_args.kwargs)
        # override_args = copy.deepcopy(base_bound_args.args)
        # override_kwargs = copy.deepcopy(base_bound_args.kwargs)

        override_kwargs = {}
        for accessor, override in zip(accessors, overrides_i):
            assert all(isinstance(part, optree.accessor.MappingEntry) for part in accessor), (
                accessor,
            )
            override_kwargs_i = override_kwargs
            for path in accessor.path[:-1]:
                override_kwargs_i = override_kwargs_i.setdefault(path, {})
            override_kwargs_i[accessor.path[-1]] = override

        override_arguments = _merge(
            override_bound_args.arguments,
            override_kwargs,
        )
        b = sig.bind_partial(
            **override_arguments,
        )
        # Create the run.
        run = _base_case(*b.args, **b.kwargs)
        runs.append(run)
    return np.array(runs).reshape(shape)


def default_run_suffix_fn(grid_pos: tuple[int, ...], grid_shape: tuple[int, ...]) -> str:
    # Option 1: _i_j_k style
    # return "_".join(map(str, grid_pos))
    # Option 2: _index style
    index = np.arange(0, np.prod(grid_shape)).reshape(grid_shape)[grid_pos]
    return f"_{index}"


def wandb_log(
    wandb_run: Run | NestedSequence[Run],
    metrics: dict[str, Any],
    step: int | np.typing.NDArray[np.integer] | np.typing.ArrayLike,
    run_index: np.typing.NDArray[np.integer] | np.typing.ArrayLike | None = None,
    metrics_are_stacked: bool | None = None,
):
    """Log metrics to wandb using `wandb.log` for each run in `wandb_run`.

    If `metrics_are_stacked` is False, the metrics are logged to every run.
    """

    wandb_run_array = np.asanyarray(wandb_run)
    multiple_runs = wandb_run_array.size > 1

    if multiple_runs and metrics_are_stacked is None:
        metrics_are_stacked = _check_shape_prefix(metrics, wandb_run_array.shape)

    # TODO: Probably won't work correctly if only one of `step` or `metrics` is traced.
    this_is_being_traced = optree.tree_any(optree.tree_map(_is_tracer, (metrics, step)))  # type: ignore
    this_is_being_vmapped = this_is_being_traced and multiple_runs and metrics_are_stacked is False
    if this_is_being_traced:
        logger.debug(
            f"Logging to wandb under a tracing context: {wandb_run_array.shape=}, "
            f"{metrics_are_stacked=}, {this_is_being_vmapped=}"
        )

    if this_is_being_vmapped:
        # Multiple wandb_runs and metrics are for a single run, this is probably being called
        # from a function that is (or is going to be?) vmapped.
        logger.debug(
            f"Assuming that the calling function is vmapped since {wandb_run_array.ndim=} and {metrics_are_stacked=}"
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
            if not _is_tracer(step):
                step = int(step.item())
                return jax.experimental.io_callback(
                    lambda _metrics: wandb_run.log(_metrics, step=step), (), metrics
                )
            if not optree.tree_all(optree.tree_map(_is_tracer, metrics)):
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
    for run_index, wandb_run_i, metrics_i in _slice(
        wandb_run_array.shape, wandb_run_array, metrics
    ):
        assert isinstance(wandb_run_i, Run)
        indexing_tuple = np.unravel_index(run_index, wandb_run_array.shape)
        # todo: re-enable this use-case: log the same metrics in all runs.
        # if not metrics_are_stacked:
        #     metrics_i = metrics
        # logger.info("Run index: %s, metrics: %s", run_index, jax.tree.map(jax.typeof, metrics))
        step_i = _get_step(step, indexing_tuple)
        log(wandb_run_i, metrics=metrics_i, step=step_i)

    return


def _get_step(
    step: int | np.typing.ArrayLike, indexing_tuple: tuple[int, ...] | tuple[np.intp, ...]
):
    if isinstance(step, int) or not hasattr(step, "shape"):
        return step
    step = typing.cast(np.typing.NDArray, step)
    if step.ndim == 0:
        if _is_tracer(step):
            # Under jax.jit we can't call .item() on a tracer.
            # The step will become an int once inside the io_callback.
            return step
        return step.item()
    assert step.ndim == len(indexing_tuple)
    return step[indexing_tuple]


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
    this_is_being_traced = optree.tree_any(optree.tree_map(_is_tracer, (wandb_run, step)))  # type: ignore

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

            assert _is_tracer(step), "assuming step is also a tracer for now."
            return jax.experimental.io_callback(
                lambda _step, *_args, **_kwargs: log(wandb_run, _step, 0, 1, *_args, **_kwargs),
                (),
                step,
                *args,
                **kwargs,
            )
        return log(wandb_run, step, 0, 1, *args, **kwargs)

    num_runs = wandb_run_array.size
    for run_index, wandb_run_i, args_i, kwargs_i in _slice(
        wandb_run_array.shape, wandb_run_array, args, kwargs
    ):
        indexing_tuple = np.unravel_index(run_index, wandb_run_array.shape)
        step_i = _get_step(step, indexing_tuple)
        if this_is_being_traced:
            import jax.experimental  # type: ignore

            assert _is_tracer(step_i), "assuming step is also a tracer for now."
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

    for run_index, wandb_run_i, args_i, kwargs_i in _slice(
        wandb_run_array.shape, wandb_run_array, args, kwargs
    ):
        indexing_tuple = np.unravel_index(run_index, wandb_run_array.shape)
        step_i = _get_step(step, indexing_tuple)
        log_fn(wandb_run_i, step_i, run_index, wandb_run_array.size, *args_i, **kwargs_i)


def _slice[*Ts](run_grid_shape: tuple[int, ...], *args: *Ts) -> Iterable[tuple[int, *Ts]]:
    """Yields the sliced args and kwargs for each run in the grid."""
    num_runs = int(np.prod(run_grid_shape))
    for run_index in range(num_runs):
        indexing_tuple = np.unravel_index(run_index, run_grid_shape)
        args_i = optree.tree_map(
            lambda v: operator.itemgetter(indexing_tuple)(v)
            if _shape_begins_with(v, run_grid_shape)
            else v,  # duplicate the metric if it doesn't have the right shape prefix?
            args,
        )  # type: ignore
        # kwargs_i = optree.tree_map(operator.itemgetter(indexing_tuple), kwargs)  # type: ignore
        yield (run_index,) + args_i


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


def _is_tracer(v: Any) -> bool:
    if "Tracer" in type(v).__name__:
        return True
    return False


def _check_shape_prefix(metrics: PyTree[Any], shape: tuple[int, ...]) -> bool:
    """Returns `True` if all the entries in `metrics` have a shape that begins with `shape`."""
    fn = functools.partial(_shape_begins_with, prefix=shape)
    return optree.tree_all(optree.tree_map(fn, metrics))


def _shape_begins_with(metric: np.typing.ArrayLike, prefix: tuple[int, ...]) -> bool:
    """Returns `True` if `metric` has a shape that begins with `prefix`."""
    if not hasattr(metric, "shape"):
        return False
    metric = typing.cast(np.typing.NDArray, metric)
    return metric.shape[: len(prefix)] == prefix


def _assert_shape_prefix[M: Mapping[str, Any]](metrics: M, shape: tuple[int, ...]) -> M:
    def _check_shape(metric: np.typing.ArrayLike):
        if not hasattr(metric, "shape"):
            return False
        metric = typing.cast(np.typing.NDArray, metric)
        if not metric.shape[: len(shape)] == shape:
            raise ValueError(
                f"Metric {metric} has shape {metric.shape}, but expected its "
                f"shape to begin with {shape}"
            )
        return metric

    return optree.tree_map(_check_shape, metrics)


def _array_first_value(array: np.typing.ArrayLike, jittable: bool = False) -> int:
    # Assume it's the same timestep for all runs, otherwise things might be tricky.
    if isinstance(array, int):
        return array
    if jittable:
        if array.ndim == 0:
            return array
        return array.flatten()[0]  # type: ignore
    else:
        if array.ndim == 0:
            return array
        step = array.flatten()[0].item()
        return step


# def getitem(v, i: int | tuple[int, ...] | slice | jax.Array | np.ndarray):
#     if isinstance(v, jax.Array):
#         # NotImplementedError: dynamic_slice on sharded dims where out dim (1) is not
#         # divisible by mesh axes (2) with spec (seed) is not implemented.
#         # No idea if this will work :(
#         return v[i]
#         return v.at[i].get(out_sharding=jax.sharding.PartitionSpec())  # type: ignore
#         # return jax.experimental.multihost_utils.process_allgather(v)[i]
#     return v[i]
