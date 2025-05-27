import os
from unittest.mock import Mock

import numpy as np
import pytest
import wandb

from parallel_wandb.init import wandb_init


@pytest.fixture
def slurm_env_vars(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Set SLURM environment variables to simulate a slurm job."""
    env_vars = getattr(request, "param", {"SLURM_JOB_ID": "12345", "SLURM_PROCID": 0})
    for var, value in env_vars.items():
        monkeypatch.setenv(var, value)
    return env_vars


@pytest.mark.parametrize(
    slurm_env_vars.__name__,
    [
        {"SLURM_JOB_ID": "12345", "SLURM_PROCID": "0"},
        {"SLURM_JOB_ID": "12345", "SLURM_PROCID": "1"},
    ],
    indirect=True,
)
def test_wandb_init_with_slurm_env_vars(slurm_env_vars: dict[str, str]):
    init = Mock(spec=wandb.init, spec_set=True)
    run = wandb_init(_wandb_init=init, project="test_project", name="test_name")
    assert run == np.asanyarray(init.return_value)
    init.assert_called_once_with(
        project="test_project",
        name="test_name",
        group=slurm_env_vars["SLURM_JOB_ID"],
        config=slurm_env_vars,
        reinit="create_new",
        **({"mode": "disabled"} if slurm_env_vars["SLURM_PROCID"] != "0" else {}),
    )


@pytest.fixture(autouse=True)
def unset_slurm_env_vars(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Unset SLURM environment variables in case tests are being run in a slurm job.

    The SLURM env vars change some defaults in the `wandb_init` function.
    """
    if (_keep_slurm_env_vars := getattr(request, "param", False)) is True:
        # By parameterizing the fixture indirectly with `True`, that signals that we want to keep
        # the SLURM env vars during a specific test.
        # This can be interesting for example when running tests inside a slurm job with
        # multiple tasks, possibly even multiple nodes!
        return
    # Temporarily unset all SLURM environment variables while tests run.
    for var in os.environ:
        if var.startswith("SLURM_"):
            monkeypatch.delenv(var, raising=False)


def test_wandb_init():
    init = Mock(spec=wandb.init, spec_set=True)
    run = wandb_init(_wandb_init=init, project="test_project", name="test_name")
    assert run == np.asanyarray(init.return_value)
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
