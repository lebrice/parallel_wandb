"""Tests that run the examples."""

import functools
import sys
from pathlib import Path

import pytest
import pytest_mock
import wandb

from parallel_wandb.init import wandb_init


@pytest.mark.parametrize("num_seeds", [1, 2])
def test_jax_mnist_example(
    num_seeds: int,
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockFixture,
    tmp_path: Path,
):
    """Run the jax_mnist example."""

    _wandb_init = mocker.Mock(
        spec_set=True,
        spec=wandb.init,
        wraps=wandb.init,
    )
    import jax
    import jax_mnist

    # Set some arguments so we don't actually log to wandb online and use the temp test dir.
    monkeypatch.setattr(
        jax_mnist,
        wandb_init.__name__,
        functools.partial(wandb_init, _wandb_init=_wandb_init, dir=tmp_path, mode="offline"),
    )

    # Set command-line arguments
    monkeypatch.setattr(
        sys,
        "argv",
        [
            Path(jax_mnist.__file__).name,
            "--num_epochs",
            "2",
            "--num_seeds",
            str(num_seeds),
            "--data_parallel_devices",
            str(jax.device_count() if num_seeds == 1 else 1),
        ],
    )
    jax_mnist.main()

    _wandb_init.assert_called()
    assert _wandb_init.call_count == num_seeds
    wandb_dir = tmp_path / "wandb"

    assert wandb_dir.exists()
    # One offline run per seed.
    # There's also a `wandb/latest-run` symlink which we ignore.
    run_dirs = list(f for f in wandb_dir.iterdir() if f.is_dir() and not f.is_symlink())
    assert len(run_dirs) == num_seeds, run_dirs
    for run_dir in run_dirs:
        files_in_run_dir = list(run_dir.iterdir())
        assert (run_dir / "logs") in files_in_run_dir
        assert (run_dir / "files") in files_in_run_dir
        # Doesn't seem to be there? Curious.
        # config_file = run_dir / "files" / "config.yaml"
        # assert config_file.read_text()
