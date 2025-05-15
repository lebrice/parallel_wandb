from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest
import wandb
from wandb.sdk.wandb_run import Run
from .log import wandb_init, wandb_log


def test_wandb_init(monkeypatch: pytest.MonkeyPatch):
    init = Mock(spec=wandb.init, spec_set=True)

    # run = wandb_init(_wandb_init=init, project="test_project", name="test_name")
    # assert run is init.return_value
    # init.assert_called_once_with(project="test_project", name="test_name", reinit="create_new")

    init.reset_mock()
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


def test_wandb_log():
    def mock_run():
        return Mock(spec=Run)

    fake_run = mock_run()
    wandb_log(fake_run, {"a": 1}, step=1)
    fake_run.log.assert_called_once_with({"a": 1}, step=1)

    fake_runs = [mock_run(), mock_run()]
    wandb_log(fake_runs, {"a": jnp.asarray([1, 2])}, step=1)
    fake_runs[0].log.assert_called_once_with({"a": 1}, step=1)
    fake_runs[1].log.assert_called_once_with({"a": 2}, step=1)

    fake_runs = [mock_run(), mock_run()]
    wandb_log(fake_runs, {"a": jnp.arange(2)}, step=jnp.asarray(1))
    fake_runs[0].log.assert_called_once_with({"a": 0}, step=1)
    fake_runs[1].log.assert_called_once_with({"a": 1}, step=1)

    fake_runs = [[mock_run(), mock_run(), mock_run()], [mock_run(), mock_run(), mock_run()]]
    wandb_log(fake_runs, {"a": jnp.arange(6).reshape(2, 3)}, step=jnp.asarray([1, 1]))
    fake_runs[0][0].log.assert_called_once_with({"a": 0}, step=1)
    fake_runs[0][1].log.assert_called_once_with({"a": 1}, step=1)
    fake_runs[0][2].log.assert_called_once_with({"a": 2}, step=1)
    fake_runs[1][0].log.assert_called_once_with({"a": 3}, step=1)
    fake_runs[1][1].log.assert_called_once_with({"a": 4}, step=1)
    fake_runs[1][2].log.assert_called_once_with({"a": 5}, step=1)
