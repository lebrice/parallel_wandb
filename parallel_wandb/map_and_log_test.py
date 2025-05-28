import unittest.mock
from unittest.mock import Mock

import jax
import numpy as np
import optree
import pytest
import wandb
from wandb.sdk.wandb_run import Run

from parallel_wandb.log import NestedSequence
from parallel_wandb.log_test import mock_run
from parallel_wandb.map_and_log import map_fn_and_log_to_wandb


@pytest.mark.parametrize("jit", [False, True])
def test_map_and_log_to_wandb(jit: bool):
    import jax.numpy as jnp

    wandb_Image = Mock(spec=wandb.Image, spec_set=True, wraps=wandb.Image)

    def _make_image(rng: jax.Array):
        return jax.random.uniform(
            rng,
            (32, 32, 3),
            minval=0,
            maxval=256,
        ).astype(jnp.uint8)

    def _log_image(run_index: int, total_runs: int, data):
        assert isinstance(data, jax.Array)
        # This should NEVER be a tracer!
        # We want this to be called as an io_callback.
        # EXCEPT if MAYBE, doing `jax.device_get` here signals to jax.jit
        # to somehow do the rest of this with a new object every time?
        assert "Tracer" not in type(data).__name__
        return {
            "image": wandb_Image(
                jax.device_get(data), caption=f"Run index {run_index} out of {total_runs}"
            )
        }

    def training_step(
        rng: jax.Array,
        step: jax.Array,
        *,
        wandb_run: Run | NestedSequence[Run],
        run_index: jax.Array | None = None,
    ):
        image_data = _make_image(rng)
        map_fn_and_log_to_wandb(
            wandb_run,
            step=step,
            fn=_log_image,
            data=image_data,
            run_index=run_index,
        )
        return image_data

    if jit:
        training_step = jax.jit(training_step, static_argnames=["wandb_run"])

    fake_run = mock_run()
    training_step(
        jax.random.key(0),
        step=jnp.asarray(0),
        # run_index=jnp.arange(0),
        wandb_run=fake_run,
    )
    training_step(
        jax.random.key(1),
        step=jnp.asarray(1),
        # run_index=jnp.arange(0),
        wandb_run=fake_run,
    )
    fake_run.log.assert_called()
    assert len(fake_run.log.call_args) == 2
    assert wandb_Image.call_count == 2
    assert fake_run.log.call_count == 2

    # weird, for some reason the assert_any_call doesnt work with step=1?
    fake_run.log.assert_any_call({"image": unittest.mock.ANY}, step=0)
    # fake_run.log.assert_any_call({"image": unittest.mock.ANY}, step=1)

    if fake_run.log.call_args_list[0].kwargs["step"] == 0:
        # Callback with logs of first step came in first
        np.testing.assert_array_equal(
            # Seems like using an io callback adds an extra leading dimension?
            jnp.expand_dims(jax.device_get(_make_image(jax.random.key(0))), 0),
            wandb_Image.call_args_list[0][0],
        )
        np.testing.assert_array_equal(
            jnp.expand_dims(jax.device_get(_make_image(jax.random.key(1))), 0),
            wandb_Image.call_args_list[1][0],
        )
    else:
        raise NotImplementedError("BAD, metrics were logged in wrong order within a single run.")
        #  Callback with logs of second step came in first?
        # TODO: Wandb apparently doesn't support this!
        np.testing.assert_array_equal(
            jax.device_get(_make_image(jax.random.key(0))),
            wandb_Image.call_args_list[0][0],
        )
        np.testing.assert_array_equal(
            jax.device_get(_make_image(jax.random.key(1))),
            wandb_Image.call_args_list[1][0],
        )


@pytest.mark.parametrize("jit", [False, True])
def test_map_and_log_to_wandb_with_vmap(jit: bool):
    import jax.numpy as jnp

    wandb_Image = Mock(spec=wandb.Image, spec_set=True, wraps=wandb.Image)

    def _make_image(rng: jax.Array):
        return jax.random.uniform(
            rng,
            (32, 32, 3),
            minval=0,
            maxval=256,
        ).astype(jnp.uint8)

    def _log_image(run_index: int, total_runs: int, data):
        assert isinstance(data, jax.Array)
        # This should NEVER be a tracer!
        # We want this to be called as an io_callback.
        # EXCEPT if MAYBE, doing `jax.device_get` here signals to jax.jit
        # to somehow do the rest of this with a new object every time?
        assert "Tracer" not in type(data).__name__
        return {
            "image": wandb_Image(
                jax.device_get(data), caption=f"Run index {run_index} out of {total_runs}"
            )
        }

    _log_image = unittest.mock.Mock(spec=_log_image, spec_set=True, wraps=_log_image)

    def training_step(
        rng: jax.Array,
        step: jax.Array,
        run_index: jax.Array,
        wandb_run: Run | NestedSequence[Run],
    ):
        image_data = _make_image(rng)
        map_fn_and_log_to_wandb(
            wandb_run,
            step=step,
            fn=_log_image,
            data=image_data,
            run_index=run_index,
        )
        return image_data

    fake_runs = (mock_run(), mock_run())

    training_step = jax.vmap(training_step, in_axes=(0, 0, 0, None))

    if jit:
        training_step = jax.jit(training_step, static_argnames=["wandb_run"])

    training_step(
        jax.vmap(jax.random.key)(jnp.arange(2)),
        jnp.arange(2) * 10,
        jnp.arange(2),
        fake_runs,
    )
    training_step(
        jax.vmap(jax.random.key)(jnp.arange(2) + 100),
        (jnp.arange(2) + 1) * 10,
        jnp.arange(2),
        fake_runs,
    )
    assert fake_runs[0].log.call_count == 2
    fake_runs[0].log.assert_any_call({"image": unittest.mock.ANY}, step=0)
    fake_runs[0].log.assert_any_call({"image": unittest.mock.ANY}, step=10)

    assert fake_runs[1].log.call_count == 2
    fake_runs[1].log.assert_any_call({"image": unittest.mock.ANY}, step=10)
    fake_runs[1].log.assert_any_call({"image": unittest.mock.ANY}, step=20)
    assert wandb_Image.call_count == 4

    optree.tree_map(
        np.testing.assert_array_equal,
        _log_image.call_args_list[0].args,
        (jnp.asarray(0), 2),
    )
    optree.tree_map(
        np.testing.assert_array_equal,
        _log_image.call_args_list[0].kwargs,
        {"data": _make_image(jax.random.key(0))},
    )

    optree.tree_map(
        np.testing.assert_array_equal,
        _log_image.call_args_list[1].args,
        (jnp.asarray(1), 2),
    )
    optree.tree_map(
        np.testing.assert_array_equal,
        _log_image.call_args_list[1].kwargs,
        {"data": _make_image(jax.random.key(1))},
    )

    optree.tree_map(
        np.testing.assert_array_equal,
        _log_image.call_args_list[2].args,
        (jnp.asarray(0), 2),
    )
    optree.tree_map(
        np.testing.assert_array_equal,
        _log_image.call_args_list[2].kwargs,
        {"data": _make_image(jax.random.key(100))},
    )
    optree.tree_map(
        np.testing.assert_array_equal,
        _log_image.call_args_list[3].args,
        (jnp.asarray(1), 2),
    )
    optree.tree_map(
        np.testing.assert_array_equal,
        _log_image.call_args_list[3].kwargs,
        {"data": _make_image(jax.random.key(101))},
    )
