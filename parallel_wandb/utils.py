import operator
import typing
from typing import Any, Iterable, Mapping, Sequence, TypeAlias, TypeVar, TypeVarTuple

import numpy as np
import optree

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
NestedSequence: TypeAlias = Sequence[T | "NestedSequence[T]"]
NestedMapping: TypeAlias = Mapping[K, V | "NestedMapping[K, V]"]


def shape_begins_with(metric: np.typing.ArrayLike, prefix: tuple[int, ...]) -> bool:
    """Returns `True` if `metric` has a shape that begins with `prefix`."""
    if not hasattr(metric, "shape"):
        return False
    metric = typing.cast(np.typing.NDArray, metric)
    return metric.shape[: len(prefix)] == prefix


Ts = TypeVarTuple("Ts")


def slice(
    run_grid_shape: tuple[int, ...], *args: *Ts, strict: bool = True
) -> Iterable[tuple[int, *Ts]]:
    """Yields the slices of `args` for each run in the grid.

    If `strict` is False and one of the args does not begin with the run_grid_shape, that arg
    is duplicated for each run index. Otherwise an error will be raised when indexing the value.
    """
    num_runs = int(np.prod(run_grid_shape))
    for run_index in range(num_runs):
        indexing_tuple = np.unravel_index(run_index, run_grid_shape)
        if strict:
            args_i = optree.tree_map(
                lambda v: operator.itemgetter(indexing_tuple)(v),
                args,
            )
        else:
            args_i = optree.tree_map(
                lambda v: operator.itemgetter(indexing_tuple)(v)
                if shape_begins_with(v, run_grid_shape)
                else v,  # duplicate the metric if it doesn't have the right shape prefix?
                args,
            )
        args_i = typing.cast(tuple[*Ts], tuple(args_i))
        yield run_index, *args_i


def is_tracer(v: Any) -> bool:
    if "Tracer" in type(v).__name__:
        return True
    return False


def get_step(
    step: int | np.typing.ArrayLike, indexing_tuple: tuple[int, ...] | tuple[np.intp, ...]
):
    if isinstance(step, int) or not hasattr(step, "shape"):
        return step
    step = typing.cast(np.typing.NDArray, step)
    if step.ndim == 0:
        if is_tracer(step):
            # Under jax.jit we can't call .item() on a tracer.
            # The step will become an int once inside the io_callback.
            return step
        return step.item()
    assert step.ndim == len(indexing_tuple)
    return step[indexing_tuple]
