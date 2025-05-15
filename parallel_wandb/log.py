"""IDEA: Create a process that when starting, does wandb.init, and when it receives things via some kind of pipe, simply calls
wandb.log.

The goal is to create a list of these subprocess handles, so that we can log to many wandb runs in parallel.
"""

import copy
import inspect
import operator
import os
from collections.abc import Callable, Sequence
from logging import getLogger
from typing import Any, Concatenate, Mapping, TypeAlias, TypeVar, TypeVarTuple
import typing

# import jax
# import jax.experimental
# import jax.experimental.multihost_utils
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
    stacked_overrides: NestedMapping[str, NestedSequence],
    process_index: int | None = None,
    _wandb_init: Callable[P, OutT] = wandb.init,
    *args: P.args,
    **kwargs: P.kwargs,
) -> NestedSequence[OutT]: ...


def wandb_init[**P, OutT](
    stacked_overrides: NestedMapping[str, NestedSequence[Any]] | None = None,
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
    if not isinstance(first_override, Sequence):
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

        _overrides = typing.cast(Any, overrides)  # typing bug in optree (list isnt pytree?)
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
    step: int | np.typing.ArrayLike,
    metrics_are_stacked: bool = True,
    jittable: bool = False,
):
    """Log metrics to wandb.

    Doesn't work under jax.jit unless `jittable` is set to True.
    """

    # Assume it's the same timestep for all runs for simplicity?
    # if not isinstance(step, int):
    #     if jittable:
    #         step = step.flatten()[0]  # type: ignore
    #     else:
    #         step = np.asarray(step).flatten()[0].item()
    #         assert isinstance(step, int)
    def _log(wandb_run: Run, metrics: dict[str, Any], step: int | np.typing.ArrayLike):
        """Base case: single run, simple dict of metrics."""
        if jittable:
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

    wandb_runs = np.asarray(wandb_run)
    num_runs = np.prod(wandb_runs.shape)
    # non-recursive version that indexes using the multi-dimensional metrics.
    for run_index in range(num_runs):
        indexing_tuple = np.unravel_index(run_index, wandb_runs.shape)
        wandb_run = wandb_runs[indexing_tuple]
        assert isinstance(wandb_run, Run)
        if not metrics_are_stacked:
            # Log the same metrics in all runs.
            metrics_i = metrics
        else:
            _metrics = typing.cast(Any, metrics)  # bug in optree.tree_map typing?
            metrics_i = optree.tree_map(operator.itemgetter(indexing_tuple), _metrics)
            metrics_i = typing.cast(dict[str, Any], metrics_i)

        _log(wandb_run, metrics_i, step=step)

    return


def map_fn_and_log_to_wandb[**P](
    wandb_run: Run | NestedSequence[Run],
    step: int | np.typing.ArrayLike,
    fn: Callable[Concatenate[tuple[int, ...], P], dict[str, Any]],
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


def _array_first_value(array: np.typing.ArrayLike, jittable: bool = False) -> int:
    # Assume it's the same timestep for all runs, otherwise things might be tricky.
    if jittable:
        return array.flatten()[0]  # type: ignore
    else:
        step = np.asarray(array).flatten()[0].item()
        assert isinstance(step, int)
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
