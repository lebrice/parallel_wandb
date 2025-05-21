# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.example_libraries.stax is for neural network building, and
the mini-library jax.example_libraries.optimizers is for first-order stochastic
optimization.
"""

import array
import functools
import gzip
import logging
import os
import struct
import sys
import time
import urllib.request
from os import path
from typing import Callable

import einops
import jax
import jax.numpy as jnp
import numpy as np
import rich.logging
from jax import NamedSharding, jit
from jax.example_libraries import optimizers, stax
from jax.example_libraries.optimizers import OptimizerState, Params
from jax.example_libraries.stax import Dense, LogSoftmax, Relu
from wandb.sdk.wandb_run import Run

from parallel_wandb.log import wandb_init, wandb_log

logger = logging.getLogger(__name__)
_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print(f"downloaded {url} to {_DATA}")


def _partial_flatten(x: np.ndarray) -> np.ndarray:
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x: np.ndarray, k: int, dtype=np.float32) -> np.ndarray:
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)  # Why?
    test_labels = _one_hot(test_labels, 10)  # Why?

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


def loss(
    params: Params,
    batch: tuple[jax.Array, jax.Array],
    predict: Callable[[Params, jax.Array], jax.Array],
):
    inputs, targets = batch
    preds = predict(params, inputs)
    neg_cross_entropy = -jnp.mean(jnp.sum(preds * targets, axis=1))
    neg_cross_entropy = jax.lax.pmean(neg_cross_entropy, axis_name="batch")
    return neg_cross_entropy, preds


def accuracy(params: Params, batch, predict: Callable):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    pred_logits = predict(params, inputs)
    predicted_class = jnp.argmax(pred_logits, axis=1)
    return jnp.mean(predicted_class == target_class)


def main():
    # Under slurm, this is perfect:
    if "SLURM_JOB_ID" in os.environ:
        jax.distributed.initialize()
    setup_logging(
        local_rank=jax.process_index(),
        num_processes=jax.process_count(),
        verbose=2,
    )
    seed = 0
    data_seed = 123
    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    rng = jax.random.key(seed)
    data_rng = jax.random.key(data_seed)
    num_seeds = 2
    data_parallel_devices = 1

    mesh = jax.make_mesh(
        (jax.device_count() // data_parallel_devices, data_parallel_devices),
        axis_names=("seed", "batch"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    seeds = seed + jnp.arange(num_seeds)
    data_seeds = data_seed + jnp.arange(data_parallel_devices)

    rngs = jax.vmap(jax.random.key)(seeds)
    data_rngs = jax.vmap(jax.random.key)(data_seeds)

    wandb_run = wandb_init(
        {"config": {"seed": seeds}},
        process_index=jax.process_index(),
        project="parallel_wandb_example",
        config=dict(
            seed=seed,
            data_seed=data_seed,
            step_size=step_size,
            num_epochs=num_epochs,
            batch_size=batch_size,
            momentum_mass=momentum_mass,
        ),
    )
    train_images, train_labels, test_images, test_labels = mnist()
    train_images, train_labels, test_images, test_labels = jax.tree.map(
        jnp.asarray, (train_images, train_labels, test_images, test_labels)
    )

    _final_states, final_test_accs = time_fn(
        jax.jit(
            jax.vmap(
                jax.vmap(
                    lambda rng, data_rng: run(
                        rng,
                        data_rng,
                        step_size=step_size,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        momentum_mass=momentum_mass,
                        train_images=train_images,
                        train_labels=train_labels,
                        test_images=test_images,
                        test_labels=test_labels,
                        wandb_run=wandb_run,
                    ),
                    in_axes=(None, 0),
                    axis_name="batch",
                ),
                in_axes=(0, None),
                axis_name="seed",
            ),
            in_shardings=(
                NamedSharding(mesh, jax.sharding.PartitionSpec("seed")),
                NamedSharding(mesh, jax.sharding.PartitionSpec("batch")),
            ),
            # out_shardings=(
            #     jax.sharding.PartitionSpec("seed"),
            #     jax.sharding.PartitionSpec("seed", "batch"),
            # ),
        )
    )(
        rngs,
        data_rngs,
    )
    print(f"Final test accuracies: {final_test_accs}")


def run(
    rng: jax.Array,
    data_rng: jax.Array,
    train_images: jax.Array,
    train_labels: jax.Array,
    test_images: jax.Array,
    test_labels: jax.Array,
    wandb_run: Run,
    step_size=0.001,
    num_epochs=10,
    batch_size=128,
    momentum_mass=0.9,
):
    num_train = train_images.shape[0]

    init_random_params, predict = stax.serial(
        Dense(1024), Relu, Dense(1024), Relu, Dense(10), LogSoftmax
    )

    num_complete_batches, _leftover = divmod(num_train, batch_size)

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)
    _, init_params = init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)

    @jit
    def update(opt_state: OptimizerState, step_and_batch: jax.Array):
        step, batch = step_and_batch
        params = get_params(opt_state)
        loss_fn = functools.partial(loss, predict=predict)
        (train_loss, preds), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        opt_state = opt_update(step, gradients, opt_state)
        labels = batch[1]
        accuracy = jnp.mean(preds.argmax(-1) == labels.argmax(-1))
        accuracy = jax.lax.pmean(accuracy, axis_name="batch")
        wandb_log(
            wandb_run,
            {"train/loss": train_loss, "train/accuracy": accuracy},
            step=step,
        )
        return opt_state, (train_loss, preds)

    @jax.jit
    def epoch(opt_state: OptimizerState, epoch: jax.Array):
        epoch_data_rng = jax.random.fold_in(data_rng, epoch)
        perm = jax.random.permutation(epoch_data_rng, num_train)
        perm = perm[: num_complete_batches * batch_size]  # drop leftover.
        epoch_train_data, epoch_train_labels = train_images[perm], train_labels[perm]
        epoch_train_data, epoch_train_labels = jax.tree.map(
            lambda v: einops.rearrange(
                v, "(n b) ... -> n b ...", n=num_complete_batches, b=batch_size
            ),
            (epoch_train_data, epoch_train_labels),
        )
        epoch_start_step = epoch * num_complete_batches
        opt_state, (train_losses, _preds) = jax.lax.scan(
            update,
            init=opt_state,
            xs=(
                # update steps
                epoch_start_step + jnp.arange(num_complete_batches),
                (epoch_train_data, epoch_train_labels),
            ),
            length=num_complete_batches,
        )
        test_loss, test_preds = loss(
            get_params(opt_state), (test_images, test_labels), predict=predict
        )
        test_accuracy = jnp.mean(test_preds.argmax(-1) == test_labels.argmax(-1))
        test_accuracy = jax.lax.pmean(test_accuracy, axis_name="batch")
        wandb_log(
            wandb_run,
            {"test/loss": test_loss, "accuracy": test_accuracy},
            step=(epoch + 1) * num_complete_batches,
        )
        return opt_state, (train_losses, test_accuracy)

    opt_state, (train_losses, test_accuracies) = jax.lax.scan(
        epoch,
        init=opt_state,
        xs=jnp.arange(num_epochs),
        length=num_epochs,
    )
    # test_acc = accuracy(get_params(opt_state), (test_images, test_labels), predict)
    jax.debug.print("Final test accuracy: {:.2%}", test_accuracies[-1])
    return opt_state, test_accuracies[-1]


def time_fn[**P, OutT](fn: Callable[P, OutT], desc: str = ""):
    desc = desc or fn.__name__

    @functools.wraps(fn)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> OutT:
        t0 = time.time()
        out = fn(*args, **kwargs)
        out = jax.block_until_ready(out)
        print(f"`{desc}` took {time.time() - t0} seconds to run.")
        return out

    return _wrapped


def setup_logging(local_rank: int, num_processes: int, verbose: int):
    # Widen the log width when running in an sbatch script.
    from parallel_wandb.log import logger

    if not sys.stdout.isatty():
        console = rich.console.Console(width=140)
    else:
        console = None
    logging.basicConfig(
        level=logging.WARNING,
        # Add the [{local_rank}/{num_processes}] prefix to log messages
        format=(
            (f"[{local_rank + 1}/{num_processes}] " if num_processes > 1 else "") + "%(message)s"
        ),
        handlers=[
            rich.logging.RichHandler(
                console=console, show_time=console is not None, rich_tracebacks=True, markup=True
            )
        ],
        force=True,
    )
    if verbose == 0:
        # logger.setLevel(logging.ERROR)
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose >= 2:
        logger.setLevel(logging.DEBUG)
    # else:
    #     assert verbose >= 3

    logging.getLogger("jax").setLevel(
        logging.DEBUG if verbose == 3 else logging.INFO if verbose == 2 else logging.WARNING
    )


if __name__ == "__main__":
    main()
