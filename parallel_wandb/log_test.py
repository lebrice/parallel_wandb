import functools
import time
from unittest.mock import Mock

import numpy as np
import optree
import pytest
from wandb.sdk.wandb_run import Run
from pathlib import Path
from parallel_wandb.init import wandb_init
from parallel_wandb.utils import NestedSequence

from .log import wandb_log

import jax
import jax.numpy as jnp


def mock_run():
    return Mock(spec=Run, disabled=False)


def test_wandb_log_single_run():
    fake_run = mock_run()
    wandb_log(fake_run, {"a": 1}, step=1)
    fake_run.log.assert_called_once_with({"a": 1}, step=1)


def test_wandb_log_multiple():
    fake_runs = [mock_run(), mock_run()]
    wandb_log(fake_runs, {"a": np.asarray([1, 2])}, step=1)
    fake_runs[0].log.assert_called_once_with({"a": 1}, step=1)
    fake_runs[1].log.assert_called_once_with({"a": 2}, step=1)


def test_wandb_log_same_metrics_in_multiple_runs():
    fake_runs = [mock_run(), mock_run()]

    # TODO: Should we let indexing errors be raised if we don't set the flag?
    with pytest.raises((IndexError, TypeError)):
        wandb_log(fake_runs, {"a": 1, "b": np.arange(2)}, step=1, same_metrics_for_all_runs=False)
        fake_runs[0].log.assert_called_once_with({"a": 1, "b": np.arange(2)}, step=1)
        fake_runs[1].log.assert_called_once_with({"a": 1, "b": np.arange(2)}, step=1)

    for _mock in fake_runs:
        _mock.reset_mock()
    wandb_log(fake_runs, {"a": np.asarray([1])}, step=1, same_metrics_for_all_runs=True)
    fake_runs[0].log.assert_called_once_with({"a": 1}, step=1)
    fake_runs[1].log.assert_called_once_with({"a": 1}, step=1)


def test_wandb_log_multiple_2d():
    fake_runs = [[mock_run(), mock_run(), mock_run()], [mock_run(), mock_run(), mock_run()]]
    wandb_log(fake_runs, {"a": np.arange(6).reshape(2, 3)}, step=np.asarray(1))
    fake_runs[0][0].log.assert_called_once_with({"a": 0}, step=1)
    fake_runs[0][1].log.assert_called_once_with({"a": 1}, step=1)
    fake_runs[0][2].log.assert_called_once_with({"a": 2}, step=1)
    fake_runs[1][0].log.assert_called_once_with({"a": 3}, step=1)
    fake_runs[1][1].log.assert_called_once_with({"a": 4}, step=1)
    fake_runs[1][2].log.assert_called_once_with({"a": 5}, step=1)


def test_wandb_log_with_different_steps_per_run():
    fake_runs = [[mock_run(), mock_run(), mock_run()], [mock_run(), mock_run(), mock_run()]]
    wandb_log(fake_runs, {"a": np.arange(6).reshape(2, 3)}, step=np.arange(10, 16).reshape(2, 3))
    fake_runs[0][0].log.assert_called_once_with({"a": 0}, step=10)
    fake_runs[0][1].log.assert_called_once_with({"a": 1}, step=11)
    fake_runs[0][2].log.assert_called_once_with({"a": 2}, step=12)
    fake_runs[1][0].log.assert_called_once_with({"a": 3}, step=13)
    fake_runs[1][1].log.assert_called_once_with({"a": 4}, step=14)
    fake_runs[1][2].log.assert_called_once_with({"a": 5}, step=15)


def test_wandb_log_with_vmap_needs_run_index_arg():
    fake_runs = [mock_run(), mock_run(), mock_run()]
    import jax
    import jax.numpy as jnp

    def _fn1(x):
        wandb_log(fake_runs, {"a": x}, step=x)
        return x + 1

    with pytest.raises(ValueError, match="need to pass the `run_index` argument"):
        _ = jax.vmap(_fn1)(jnp.arange(3))


def test_wandb_log_with_vmap():
    fake_runs = [mock_run(), mock_run(), mock_run()]
    import jax
    import jax.numpy as jnp

    def _fn(x, run_index: jax.Array):
        wandb_log(fake_runs, {"a": x}, step=x, run_index=run_index)
        return x + 1

    outs = jax.vmap(_fn)(jnp.arange(3) * 10, run_index=jnp.arange(3))
    np.testing.assert_array_equal(outs, (jnp.arange(3) * 10) + 1)
    fake_runs[0].log.assert_called_once_with({"a": jnp.array(0, dtype=jnp.int32)}, step=0)
    fake_runs[1].log.assert_called_once_with({"a": jnp.array(10, dtype=jnp.int32)}, step=10)
    fake_runs[2].log.assert_called_once_with({"a": jnp.array(20, dtype=jnp.int32)}, step=20)


def training_step(
    carry: jax.Array,
    input: jax.Array,
    *,
    wandb_run: Run | NestedSequence[Run],
    run_index: jax.Array | None = None,
):
    output = carry**2 + input
    wandb_log(
        wandb_run,
        {"input": input, "output": output, "carry": carry},
        step=carry,
        run_index=run_index,
    )
    return carry + 1, output


def test_log_in_scan(tmp_path: Path):
    # Single run:
    initial_value = 0
    inputs = jnp.arange(5)
    wandb_run = wandb_init(
        {"config": {"initial_value": [initial_value]}},
        project="test_project",
        group="testing",
        dir=tmp_path,
        mode="offline",
    )
    assert isinstance(wandb_run, np.ndarray)
    wandb_run = np.asanyarray(Mock(wraps=wandb_run.tolist()[0]))

    final_value, output = jax.lax.scan(
        functools.partial(training_step, wandb_run=wandb_run, run_index=jnp.arange(1)),
        initial_value,
        inputs,
    )
    jax.block_until_ready((final_value, output))
    time.sleep(0.1)  # let teh callbacks run?
    wandb_run.item().log.assert_called()


def test_log_in_scan_with_vmap(tmp_path: Path):
    initial_values = jnp.arange(3)
    wandb_run = wandb_init(
        {"config": {"initial_value": initial_values}},
        project="test_project",
        group="testing",
        mode="offline",
        dir=tmp_path,
    )
    assert isinstance(wandb_run, np.ndarray)
    wandb_run = tuple(wandb_run.tolist())
    wandb_run = optree.tree_map(lambda r: Mock(wraps=r), wandb_run)
    inputs = jnp.arange(5)
    final_values, outputs = jax.lax.scan(
        jax.vmap(
            functools.partial(
                training_step, wandb_run=wandb_run, run_index=jnp.arange(len(initial_values))
            ),
            in_axes=(0, None),
        ),
        initial_values,
        inputs,
    )
    jax.block_until_ready((final_values, outputs))
    assert False, [wandb_run[i].log.mock_calls for i in range(len(wandb_run))]
