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
import itertools
import os
import struct
import time
from typing import Callable
import urllib.request
from os import path

import einops
import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
from jax import jit, random
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, LogSoftmax, Relu
from parallel_wandb.log import wandb_init, wandb_log
from jax.example_libraries.optimizers import OptimizerState, Params

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
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


def loss(
    params: Params,
    batch: tuple[jax.Array, jax.Array],
    predict: Callable[[jax.Array, jax.Array], jax.Array],
):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1)), preds


def accuracy(params: Params, batch, predict: Callable):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    pred_logits = predict(params, inputs)
    predicted_class = jnp.argmax(pred_logits, axis=1)
    return jnp.mean(predicted_class == target_class)


def main():
    seed = 0
    data_seed = 123
    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    wandb_run = wandb_init(
        project="parallel_wandb_example",
        config=dict(
            step_size=step_size,
            num_epochs=num_epochs,
            batch_size=batch_size,
            momentum_mass=momentum_mass,
            data_seed=data_seed,
        ),
    )

    init_random_params, predict = stax.serial(
        Dense(1024), Relu, Dense(1024), Relu, Dense(10), LogSoftmax
    )
    rng = random.key(seed)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images, train_labels, test_images, test_labels = jax.tree.map(
        jnp.asarray, (train_images, train_labels, test_images, test_labels)
    )
    num_train = train_images.shape[0]

    num_complete_batches, leftover = divmod(num_train, batch_size)

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)
    _, init_params = init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)
    data_rng = jax.random.key(data_seed)

    @jit
    def update(opt_state: OptimizerState, i_batch):
        i, batch = i_batch
        params = get_params(opt_state)
        loss_fn = functools.partial(loss, predict=predict)
        (loss_i, preds), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        opt_state = opt_update(i, gradients, opt_state)
        return opt_state, (loss_i, preds)

    @jax.jit
    def epoch(opt_state, epoch: int):
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
        opt_state, (losses, preds) = jax.lax.scan(
            update,
            init=opt_state,
            xs=(jnp.arange(num_complete_batches), (epoch_train_data, epoch_train_labels)),
            length=num_complete_batches,
        )
        accuracies = jnp.mean(preds.argmax(axis=-1) == epoch_train_labels.argmax(axis=-1))
        wandb_log(wandb_run, {"loss": losses.mean(), "accuracy": accuracies}, step=epoch)
        return opt_state, (losses, accuracies)

    # for epoch_i in range(num_epochs):
    #     # batch_idx = perm[i * batch_size : (i + 1) * batch_size]
    #     # yield train_images[batch_idx], train_labels[batch_idx]
    #     start_time = time.time()
    #     opt_state, (losses, avg_train_acc) = epoch(opt_state, epoch_i)
    #     opt_state = jax.block_until_ready(opt_state)
    #     epoch_time = time.time() - start_time
    #     print(f"Epoch {epoch_i} in {epoch_time:0.2f} sec")

    #     print(f"Training average accuracy: {avg_train_acc:.4f}")
    #     params = get_params(opt_state)
    #     test_acc = accuracy(params, (test_images, test_labels), predict)
    #     print(f"Test set accuracy {test_acc}")

    # train_acc = accuracy(params, (train_images, train_labels), predict)
    # print(f"Training set accuracy {train_acc}")

    opt_state, (train_losses, avg_train_accuracies) = jax.lax.scan(
        epoch,
        init=opt_state,
        xs=jnp.arange(num_epochs),
        length=num_epochs,
    )


if __name__ == "__main__":
    main()
