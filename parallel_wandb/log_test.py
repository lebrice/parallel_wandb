from unittest.mock import Mock

import numpy as np
import pytest
from wandb.sdk.wandb_run import Run

from .log import wandb_log


def mock_run():
    return Mock(spec=Run)


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
    wandb_log(fake_runs, {"a": np.asarray([1])}, step=1, metrics_are_stacked=False)
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
