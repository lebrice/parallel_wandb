"""IDEA: Create a process that when starting, does wandb.init, and when it receives things via some kind of pipe, simply calls
wandb.log.

The goal is to create a list of these subprocess handles, so that we can log to many wandb runs in parallel.
"""

import inspect
import operator
import os
from collections.abc import Callable, Sequence
from logging import getLogger
from typing import Any, Concatenate, Mapping, TypeVar
import typing

import numpy as np
import optree
import optree.accessor
import wandb
from wandb.sdk.wandb_run import Run

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
type NestedSequence[T] = Sequence[T | NestedSequence[T]]
type NestedMapping[K, V] = Mapping[K, V | NestedMapping[K, V]]

logger = getLogger(__name__)


# IDEA: only show handles to Jax, put the Run objects in a global variable. (ugly)
# RUN_OBJECTS: dict[int, Run] = {}


@typing.overload
def wandb_init[**P, OutT](
    stacked_overrides: None = None,
    process_index: int | None = None,
    _wandb_init: Callable[P, OutT] = wandb.init,
    *args: P.args,
    **kwargs: P.kwargs,
) -> OutT: ...


@typing.overload
def wandb_init[**P, OutT](
    stacked_overrides: NestedMapping[str, np.typing.ArrayLike],
    process_index: int | None = None,
    _wandb_init: Callable[P, OutT] = wandb.init,
    *args: P.args,
    **kwargs: P.kwargs,
) -> NestedSequence[OutT]: ...


def wandb_init[**P, OutT](
    stacked_overrides: NestedMapping[str, np.typing.ArrayLike] | None = None,
    process_index: int | None = None,
    _wandb_init: Callable[P, OutT] = wandb.init,
    *args: P.args,
    **kwargs: P.kwargs,
) -> OutT | NestedSequence[OutT]:
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

    For example:

    ```python
    wandb_init({"name": [["run_1", "run_2"], ["run_3", "run_4]], "config": {"seed": [[1, 2], [3, 4]]}})
    # This will create three runs like so:
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

    def _base_case(*args: P.args, **kwargs: P.kwargs) -> OutT:
        kwargs["reinit"] = "create_new"  # Essential: Makes it possible to create multiple runs.
        return _wandb_init(*args, **kwargs)

    if not stacked_overrides:
        return _base_case(*args, **kwargs)

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


def default_run_suffix_fn(grid_pos: tuple[int, ...], grid_shape: tuple[int, ...]) -> str:
    # Option 1: _i_j_k style
    # return "_".join(map(str, grid_pos))
    # Option 2: _index style
    index = np.arange(0, np.prod(grid_shape)).reshape(grid_shape)[grid_pos]
    return f"_{index}"


def wandb_log(
    wandb_run: Run | NestedSequence[Run],
    metrics: dict[str, Any],
    # run_index: tuple[int, ...] | None = None,
    step: int | np.typing.NDArray[np.integer] | np.typing.ArrayLike,
    run_index: np.typing.NDArray[np.integer] | np.typing.ArrayLike | None = None,
    metrics_are_stacked: bool | None = None,
    calling_fn_will_be_jitted: bool | None = None,
    calling_fn_will_be_vmapped: bool | None = None,
):
    """Log metrics to wandb.

    Doesn't work under jax.jit unless `jittable` is set to True.
    """
    wandb_run_array = np.asanyarray(wandb_run)

    if calling_fn_will_be_jitted is None:
        # TODO: Do we need the `step` and `metrics` to both be (or not be) traced?
        # If not, then we should probably adapt the code below to handle the case where step is not a tracer,
        # otherwise we might get some errors I think.
        calling_fn_will_be_jitted = optree.tree_all(optree.tree_map(_is_tracer, (metrics, step)))
        logger.debug(f"Calling function will be jitted: {calling_fn_will_be_jitted}")

    if metrics_are_stacked is None:
        metrics_are_stacked = _check_shape_prefix(metrics, wandb_run_array.shape)

    if calling_fn_will_be_vmapped is None:
        if wandb_run_array.ndim >= 1 and not metrics_are_stacked:
            # Multiple wandb_runs and metrics are for a single run, this is probably being called
            # from a function that is (or is going to be?) vmapped.
            logger.debug(
                f"Assuming that the calling function will be vmapped since {wandb_run_array.ndim=} and {metrics_are_stacked=}"
            )
            calling_fn_will_be_vmapped = True

    if calling_fn_will_be_vmapped:
        if isinstance(wandb_run, Run):
            raise ValueError(
                f"When `{calling_fn_will_be_vmapped=}` (because is is inferred to be True "
                f"or explicitly passed as an argument), then we expect `wandb_run` to be a "
                f"(possibly-nested) sequence of Wandb Run objects!"
            )
        # There are multiple wandb runs, and metrics are not stacked
        # (dont have the wandb_runs shape as a prefix in their shapes)
        # --> This is probably being called inside a function that is being vmapped!
        if run_index is None:
            raise ValueError(
                f"There are multiple wandb runs, and metrics are not stacked "
                f"(they dont have the {wandb_run_array.shape=} as a prefix in their shapes). \n"
                f"This means that you are likely calling `{wandb_log.__name__}` inside a function "
                f"that is being vmapped!\n"
                f"In this case, and since we can't tell at which 'index' in the vmap we're at, "
                f"you also need to pass the `run_index` argument. "
                f"This will be used to index into the `wandb_runs` array to select which run to log at.\n"
                f"`run_index=jnp.arange(num_seeds)` is a good option.\n"
                f"See the `jax_mnist.py` file example in the `parallel_wandb` repo for an example.\n"
                f"Metric shapes: {optree.tree_map(operator.attrgetter('shape'), metrics)}"
            )
        return wandb_log_under_vmap(wandb_run, run_index=run_index, metrics=metrics, step=step)

    # Assume it's the same timestep for all runs for simplicity?
    # if not isinstance(step, int):
    #     if jittable:
    #         step = step.flatten()[0]  # type: ignore
    #     else:
    #         step = np.asarray(step).flatten()[0].item()
    #         assert isinstance(step, int)
    def _log(wandb_run: Run, metrics: dict[str, Any], step: int | np.typing.ArrayLike):
        """Base case: single run, simple dict of metrics."""
        if calling_fn_will_be_jitted:
            import jax.experimental  # type: ignore

            # IDEA: use regular wandb_run.log when not in a jit context?
            # import jax.core  # type: ignore
            # _metrics = typing.cast(Any, metrics)  # bug in optree.tree_map typing?
            # if not optree.tree_any(optree.tree_map(lambda v: isinstance(v, jax.core.Tracer), (_metrics, step))):
            #     # No need to use an external callback: apparently not under Jit context!
            #     wandb_run.log(metrics, step=step)

            # IDEA: Try using the sharding argument to io_callback to only log from the first device?
            return jax.experimental.io_callback(wandb_run.log, (), metrics, step=step)
        if isinstance(step, np.ndarray) or (
            hasattr(step, "ndim") and callable(getattr(step, "item", None))
        ):
            assert step.ndim == 0, step  # type: ignore
            step = step.item()  # type: ignore
        assert isinstance(step, int), step
        return wandb_run.log(metrics, step=step)

    if isinstance(wandb_run, Run):
        return _log(wandb_run, metrics, step)

    wandb_runs = np.asanyarray(wandb_run)
    num_runs = np.prod(wandb_runs.shape)

    def _check_shape(metric: Any):
        if not metric.shape[: len(wandb_runs.shape)] == wandb_runs.shape:
            raise ValueError(
                f"Metric {metric} has shape {metric.shape}, but expected its shape to begin with {wandb_runs.shape}"
            )
        return metric

    metrics = optree.tree_map(_check_shape, metrics)

    # non-recursive version that indexes using the multi-dimensional metrics.
    for run_index in range(num_runs):
        indexing_tuple = np.unravel_index(run_index, wandb_runs.shape)
        wandb_run = wandb_runs[indexing_tuple]
        assert isinstance(wandb_run, Run)
        if not metrics_are_stacked:
            # Log the same metrics in all runs.
            metrics_i = metrics
        else:
            # logger.info("Run index: %s, metrics: %s", run_index, jax.tree.map(jax.typeof, metrics))
            # return
            _metrics_i = typing.cast(Any, metrics_i)  # bug in optree.tree_map typing?
            metrics_i = optree.tree_map(operator.itemgetter(indexing_tuple), _metrics_i)
            metrics_i = typing.cast(dict[str, Any], metrics_i)

        step_i = (
            step
            if isinstance(step, int) or (step.ndim == 0)
            else step[indexing_tuple]
            if step.ndim == len(indexing_tuple)
            else step[run_index]
            if step.ndim == 1
            else _array_first_value(step, jittable=calling_fn_will_be_jitted)
        )
        _log(wandb_run, metrics_i, step=step_i)

    return


def wandb_log_under_vmap(
    wandb_run: NestedSequence[Run],
    run_index: np.typing.NDArray[np.integer],
    metrics: dict[str, Any],
    step: np.typing.NDArray[np.integer],
):
    """WIP: Call to wandb.log inside a function that is vmapped, such as a `train_step`-esque function.

    In this scenario:
    - This function is being vmapped to train multiple runs in parallel.
    - wandb_run is an array of wandb runs
    - `metrics` is a dictionary of metrics to log, but it is NOT stacked!
        - We're only seeing things from the perspective of a single run! (TODO: Unclear why exactly)
    - We don't know which "run index" we're in, unless `run_index` is passed in.
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
    # We're inside vmap, so we should only have the metrics for a single run (really?)
    assert not _check_shape_prefix(metrics, wandb_run_array.shape)

    jax.experimental.io_callback(
        log,
        (),
        metrics,
        step=step,
        run_index=run_index,
    )

    # wandb_log(
    #     wandb_run,
    #     {"train/loss": train_loss, "train/accuracy": accuracy},
    #     step=step,
    # )


def map_fn_and_log_to_wandb[**P](
    wandb_run: Run | NestedSequence[Run],
    step: int | np.typing.ArrayLike,
    fn: Callable[Concatenate[tuple[int, ...], P], dict[str, Any]],
    jittable: bool = False,
    *args: P.args,
    **kwargs: P.kwargs,
):
    """Map a function over the (sliced) arg and kwargs and log the results to wandb.

    `fn` should be a function that takes a grid position (tuple of ints) in addition
    to args and kwargs, then return a dictionary of stuff to log to wandb.

    - If `wandb_run` is a single run, the function will be called with an empty
      tuple as first argument and the args and kwargs unchanged.
    - If `wandb_run` is a list of runs, the function will be called with the
      current position in the grid as the first argument, followed by the sliced
      args and kwargs.

    This works recursively, so the `wandb_run` can be a list of list of wandb runs, etc.
    """
    # Assume it's the same timestep for all runs, otherwise things might be tricky.
    if not isinstance(step, int):
        step = _array_first_value(step, jittable=jittable)

    def log_fn[**P2](
        wandb_run: Run | NestedSequence[Run],
        grid_pos: tuple[int, ...],
        fn: Callable[Concatenate[tuple[int, ...], P2], dict[str, Any]],
        *args: P2.args,
        **kwargs: P2.kwargs,
    ):
        if not isinstance(wandb_run, Sequence):
            metrics = fn(grid_pos, *args, **kwargs)
            # Base case: single run, single metric.
            logger.debug(
                "Logging to wandb run %s: metrics=%s step=%s", wandb_run.name, metrics, step
            )
            wandb_run.log(metrics, step=step)
            return
        for i, wandb_run in enumerate(wandb_run):
            # operator.itemgetter(i)
            args_i = optree.tree_map(operator.itemgetter(i), args)
            kwargs_i = optree.tree_map(operator.itemgetter(i), kwargs)
            # Recurse
            log_fn(wandb_run, grid_pos + (i,), fn, *args_i, **kwargs_i)

    log_fn(wandb_run, (), fn, *args, **kwargs)


def _is_tracer(v: Any) -> bool:
    if "Tracer" in type(v).__name__:
        return True
    return False


def _check_shape_prefix[M: Mapping[str, Any]](metrics: M, shape: tuple[int, ...]) -> bool:
    """Returns `True` if all the entries in `metrics` have a shape that begins with `shape`."""

    def _check_shape(metric: np.typing.ArrayLike):
        if not hasattr(metric, "shape"):
            return False
        metric = typing.cast(np.typing.NDArray, metric)
        return metric.shape[: len(shape)] == shape

    return optree.tree_all(optree.tree_map(_check_shape, metrics))


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
