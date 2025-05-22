from unittest.mock import Mock

import pytest
import wandb
from wandb.sdk.wandb_run import Run
from .log import wandb_init, wandb_log
import numpy as np


def test_wandb_init():
    init = Mock(spec=wandb.init, spec_set=True)
    run = wandb_init(_wandb_init=init, project="test_project", name="test_name")
    assert run is init.return_value
    init.assert_called_once_with(project="test_project", name="test_name", reinit="create_new")


def test_wandb_init_multiple():
    init = Mock(spec=wandb.init, spec_set=True)
    runs = wandb_init(
        {"name": ["run_0", "run_1"], "config": {"seed": [0, 1]}},
        _wandb_init=init,
        project="test_project",
        name="test_name",
    )
    assert isinstance(runs, np.ndarray) and runs.shape == (2,) and runs.dtype == object
    init.assert_any_call(
        name="run_0", project="test_project", config={"seed": 0}, reinit="create_new"
    )
    init.assert_any_call(
        name="run_1", project="test_project", config={"seed": 1}, reinit="create_new"
    )
    assert init.call_count == 2


def test_wandb_init_multiple_with_config():
    init = Mock(spec=wandb.init, spec_set=True)
    run = wandb_init(
        {"config": {"seed": [1, 2, 3]}},
        _wandb_init=init,
        project="test_project",
        name="test_name",
        config={"bob": 1},
    )
    assert isinstance(run, np.ndarray) and run.dtype == object
    assert run.shape == (3,)
    init.assert_any_call(
        name="test_name", project="test_project", config={"seed": 1, "bob": 1}, reinit="create_new"
    )
    init.assert_any_call(
        name="test_name", project="test_project", config={"seed": 2, "bob": 1}, reinit="create_new"
    )
    init.assert_any_call(
        name="test_name", project="test_project", config={"seed": 3, "bob": 1}, reinit="create_new"
    )


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


def test_wandb_log_with_vmap():
    # TODO:
    fake_runs = [mock_run(), mock_run(), mock_run()]
    import jax
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="need to pass the `run_index` argument"):

        def _fn(x):
            wandb_log(fake_runs, {"a": x}, step=x)
            return x + 1

        outs = jax.vmap(_fn)(jnp.arange(3))
        np.testing.assert_array_equal(outs, jnp.arange(1, 4))
        fake_runs[0].log.assert_called_once_with({"a": 0}, step=0)
        fake_runs[1].log.assert_called_once_with({"a": 1}, step=1)
        fake_runs[2].log.assert_called_once_with({"a": 2}, step=2)

    def _fn(x, run_index: jax.Array):
        wandb_log(fake_runs, {"a": x}, step=x, run_index=run_index)
        return x + 1

    outs = jax.vmap(_fn)(jnp.arange(3) * 10, run_index=jnp.arange(3))
    np.testing.assert_array_equal(outs, (jnp.arange(3) * 10) + 1)
    fake_runs[0].log.assert_called_once_with({"a": jnp.array(0, dtype=jnp.int32)}, step=0)
    fake_runs[1].log.assert_called_once_with({"a": jnp.array(10, dtype=jnp.int32)}, step=10)
    fake_runs[2].log.assert_called_once_with({"a": jnp.array(20, dtype=jnp.int32)}, step=20)
