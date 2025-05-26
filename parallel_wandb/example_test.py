"""Tests that run the examples."""

import functools
import sys
from pathlib import Path

import pytest
import pytest_mock
import wandb

from parallel_wandb.init import wandb_init


def test_jax_mnist_example(
    monkeypatch: pytest.MonkeyPatch, mocker: pytest_mock.MockFixture, tmp_path: Path
):
    """Run the jax_mnist example."""
    monkeypatch.setenv("WANDB_MODE", "offline")  # Avoid actual online WandB logging during tests
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    _wandb_init = mocker.Mock(spec_set=True, spec=wandb.init, wraps=wandb.init)
    # mocker.patch("wandb.init", return_value=_wandb_init)
    import jax_mnist

    # TODO: Problem is that this is stuck as the default in the wandb_init signature,
    # so patching wandb.init or the module attribute has no effect!
    monkeypatch.setattr(
        jax_mnist,
        wandb_init.__name__,
        functools.partial(wandb_init, _wandb_init=_wandb_init),
    )

    # Set command-line arguments

    num_seeds = 4
    monkeypatch.setattr(
        sys,
        "argv",
        [Path(jax_mnist.__file__).name, "--num_epochs", "2", "--num_seeds", str(num_seeds)],
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
        config_file = run_dir / "files" / "config.yaml"
        assert False, config_file.read_text()
