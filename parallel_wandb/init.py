# IDEA: only show handles to Jax, put the Run objects in a global variable. (ugly)
# RUN_OBJECTS: dict[int, Run] = {}


import inspect
import operator
import os
import typing
from collections.abc import Callable, Sequence
from typing import Any, ParamSpec

import numpy as np
import optree
import optree.accessor
import wandb
from wandb.sdk.wandb_run import Run

from parallel_wandb.log import _merge
from parallel_wandb.utils import NestedMapping, NestedSequence, is_tracer

P = ParamSpec("P")


def wandb_init(
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
    if optree.tree_any(optree.tree_map(is_tracer, (stacked_overrides, args, kwargs))):  # type: ignore
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

    # Not binding to `_wandb_init` directly to more easily use a mock in tests.
    sig = inspect.signature(wandb.init)
    base_bound_args = sig.bind_partial(*args, **kwargs)

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
        assert "kwargs" not in override_arguments, override_arguments
        b = sig.bind_partial(
            **override_arguments,
        )
        # Create the run.
        run = _base_case(*b.args, **b.kwargs)
        runs.append(run)
    return np.array(runs).reshape(shape)
