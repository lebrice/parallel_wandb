"""An example of parallel wandb logging.

This example is adapted from https://github.com/google/flax/blob/main/examples/mnist/train.py


A basic MNIST example using JAX with the mini-libraries stax and optimizers.

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
from dataclasses import dataclass
from os import path
from typing import Callable

import einops
import jax
import jax.numpy as jnp
import numpy as np
import rich.logging
import rich.pretty
import simple_parsing
import wandb
from jax import NamedSharding
from jax.example_libraries import optimizers, stax
from jax.example_libraries.optimizers import OptimizerState, Params
from jax.example_libraries.stax import Dense, LogSoftmax, Relu
from wandb.sdk.wandb_run import Run

from parallel_wandb.init import wandb_init
from parallel_wandb.log import NestedSequence, wandb_log
from parallel_wandb.map_and_log import map_fn_and_log_to_wandb

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


logger = logging.getLogger(__name__)
_DATA = "/tmp/jax_example_data/"


@dataclass
class Args:
    seed: int = 0
    data_seed: int = 123
    step_size: float = 0.001
    num_epochs: int = 10
    batch_size: int = 128
    momentum_mass: float = 0.9
    num_seeds: int = 1
    data_parallel_devices: int = 1
    args_are_per_device: bool = False
    jit: bool = True


def main():
    # Under slurm, this is perfect:
    if "SLURM_JOB_ID" in os.environ and "SLURM_STEP_NODELIST" in os.environ:
        jax.distributed.initialize()
    setup_logging(
        local_rank=jax.process_index(),
        num_processes=jax.process_count(),
        verbose=2,
    )
    logger.info(f"{jax.local_devices()=}, {jax.devices()=}")

    args = simple_parsing.parse(Args)

    seed = args.seed
    data_seed = args.data_seed
    step_size = args.step_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    momentum_mass = args.momentum_mass
    num_seeds = args.num_seeds
    data_parallel_devices = args.data_parallel_devices
    args_are_per_device = args.args_are_per_device

    if (n_devices := jax.device_count()) > 1 and num_seeds == 1 and data_parallel_devices == 1:
        raise RuntimeError(
            f"{num_seeds=} and {data_parallel_devices=}, but there are {n_devices} devices available. "
            f"Either:\n"
            f"  1) setting '--num_seeds' to a multiple of {n_devices=} so that each device\n"
            f"     is used for at least one run, OR:\n"
            f"  2) setting '--data_parallel_devices' to {n_devices} to use all devices\n"
            f"     for a single data-parallel training run."
        )

    if not args_are_per_device and data_parallel_devices > 1:
        # Note: for now here we assume that the given batch size is meant to be spread across devices,
        # (so that we ideally get the same result with one or two devices for the same batch size)
        # Another option could be to assume that `batch_size` is the size per device, and leave it as-is.
        # This would multiply the effective batch size by the number of devices.
        global_batch_size = batch_size
        batch_size = batch_size // data_parallel_devices
        logger.warning(
            f"Assuming that original `--batch_size` value ({global_batch_size}) represents "
            f"the global batch size. Setting per-device batch size to {batch_size}."
        )

    mesh = jax.make_mesh(
        (jax.device_count() // data_parallel_devices, data_parallel_devices),
        axis_names=("seed", "batch"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )

    seeds = seed + jnp.arange(num_seeds)

    rngs = jax.vmap(jax.random.key)(seeds)
    data_rngs = jax.vmap(jax.random.key)(data_seed + jnp.arange(data_parallel_devices))
    # data_rngs = jnp.tile(jnp.expand_dims(data_rngs, 0), (num_seeds, 1))

    wandb_run = wandb_init(
        {"name": [f"{seed=}" for seed in seeds.tolist()], "config": {"seed": seeds}},
        process_index=jax.process_index(),
        project="parallel_wandb_example",
        group=(
            f"{os.environ['SLURM_JOB_ID']}_{os.environ['SLURM_STEP_ID']}"
            if "SLURM_JOB_ID" in os.environ and "SLURM_STEP_ID" in os.environ
            else None
        ),
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
    run_fn = functools.partial(
        run,
        # rng,
        # data_rng,
        # run_index,
        step_size=step_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        momentum_mass=momentum_mass,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        wandb_run=wandb_run,
    )
    data_parallel_run_fn = jax.vmap(
        run_fn,
        in_axes=(None, 0, None),
        axis_name="batch",
    )
    multi_seed_data_parallel_run_fn = jax.vmap(
        data_parallel_run_fn,
        in_axes=(0, None, 0),
        axis_name="seed",
    )
    if args.jit:
        # Use JIT's `in_shardings` to let jax distribute the data between devices.
        jitted_sharded_run_fn = jax.jit(
            multi_seed_data_parallel_run_fn,
            in_shardings=(
                NamedSharding(mesh, jax.sharding.PartitionSpec("seed")),
                NamedSharding(mesh, jax.sharding.PartitionSpec("batch")),
                NamedSharding(mesh, jax.sharding.PartitionSpec("seed")),
            ),
            # Replicate the outputs on all devices:
            out_shardings=NamedSharding(mesh, jax.sharding.PartitionSpec()),
        )
    else:
        jitted_sharded_run_fn = multi_seed_data_parallel_run_fn
    final_state, final_test_accs = _time_fn(jitted_sharded_run_fn, desc="run")(
        rngs, data_rngs, jnp.arange(num_seeds)
    )
    logger.info(
        f"Final state structure:\n{rich.pretty.pretty_repr(jax.tree.map(jax.typeof, final_state))}"
    )
    print(f"{final_test_accs=}")


def run(
    rng: jax.Array,
    data_rng: jax.Array,
    run_index: jax.Array,
    train_images: jax.Array,
    train_labels: jax.Array,
    test_images: jax.Array,
    test_labels: jax.Array,
    wandb_run: Run | NestedSequence[Run],
    step_size: float | jax.Array = 0.001,
    num_epochs: int = 10,
    batch_size: int = 128,
    momentum_mass: float | jax.Array = 0.9,
):
    num_train_images = train_images.shape[0]
    num_train_batches = num_train_images // batch_size

    # Create the network structure:
    # - `init_random_params` is a function that takes a random key and input shape
    #    and returns the initial (random) parameters.
    # - `predict` is a function that takes parameters and inputs and returns
    #    the predicted outputs.
    init_random_params, predict = stax.serial(
        stax.Flatten, Dense(1024), Relu, Dense(1024), Relu, Dense(10), LogSoftmax
    )

    # Create the optimizer:
    # - `opt_init` initializes the optimizer state by 'wrapping' the network parameters
    #    creating optimizer-specific buffers.
    # - `opt_update` updates the parameters and optimizer state.
    # - `get_params` extracts just the model parameters portion of the "optimizer state".
    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

    def update(
        opt_state: OptimizerState,
        step_and_batch: tuple[jax.Array, tuple[jax.Array, jax.Array]],
    ):
        step, (inputs, targets) = step_and_batch
        params = get_params(opt_state)
        # Use this network when computing the loss:
        loss_fn = functools.partial(loss, predict=predict)
        (train_loss, preds), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            params, inputs, targets
        )
        opt_state = opt_update(step, gradients, opt_state)  # type: ignore (step is array, which is fine).
        accuracy = get_accuracy(preds, targets)
        accuracy = jax.lax.pmean(accuracy, axis_name="batch")

        wandb_log(
            wandb_run,
            metrics={"train/loss": train_loss, "train/accuracy": accuracy},
            step=step,
            run_index=run_index,
        )
        return opt_state, (train_loss, preds)

    def epoch(opt_state: OptimizerState, epoch: jax.Array):
        # Get a different random key at each epoch (which is used to shuffle the data).
        epoch_data_rng = jax.random.fold_in(data_rng, epoch)

        perm = jax.random.permutation(epoch_data_rng, num_train_images)
        perm = perm[: num_train_batches * batch_size]  # drop leftover indices.
        # Shuffle the training images and labels.
        epoch_train_data, epoch_train_labels = train_images[perm], train_labels[perm]

        # Rearrange the training data into batches.
        epoch_train_data, epoch_train_labels = jax.tree.map(
            lambda v: einops.rearrange(v, "(n b) ... -> n b ...", b=batch_size),
            (epoch_train_data, epoch_train_labels),
        )
        assert isinstance(epoch_train_data, jax.Array)
        assert isinstance(epoch_train_labels, jax.Array)
        epoch_start_step = epoch * num_train_batches
        opt_state, (_train_losses, _preds) = jax.lax.scan(
            update,
            init=opt_state,
            xs=(
                epoch_start_step + jnp.arange(num_train_batches),
                (epoch_train_data, epoch_train_labels),
            ),
            length=num_train_batches,
        )
        # note: forward pass with huge batch size (entire test set). Not ideal!
        params = get_params(opt_state)
        test_preds = predict(params, test_images)
        test_loss = cross_entropy_loss(test_preds, test_labels)
        test_accuracy = get_accuracy(test_preds, test_labels)

        test_loss = jax.lax.pmean(test_loss, axis_name="batch")
        test_accuracy = jax.lax.pmean(test_accuracy, axis_name="batch")
        n_rows_in_table = 20
        map_fn_and_log_to_wandb(
            wandb_run,
            fn=lambda run_index, num_runs, images, targets, predicted, probabilities: {
                "test_prediction": wandb.Table(
                    columns=[
                        "id",
                        "image",
                        "guess",
                        "truth",
                        *("score_" + str(i) for i in range(10)),
                    ],
                    data=[
                        [
                            i,
                            wandb.Image(image, classes=[targets[i]]),
                            predicted[i],
                            targets[i],
                            *probabilities[i],
                        ]
                        for i, image in enumerate(images)
                    ],
                ),
            },
            step=(epoch + 1) * num_train_batches,
            run_index=run_index,
            images=einops.rearrange(test_images[:n_rows_in_table], "n h w -> n 1 h w"),
            targets=test_labels[:n_rows_in_table],
            # outputs=test_preds[:n_rows_in_table],
            predicted=test_preds[:n_rows_in_table].argmax(-1),
            # network outputs come from log_softmax
            probabilities=jnp.exp(test_preds[:n_rows_in_table]),
            # predictions=test_preds[:20],
            # targets=test_labels[:20],
        )
        wandb_log(
            wandb_run,
            {"test/loss": test_loss, "accuracy": test_accuracy},
            step=(epoch + 1) * num_train_batches,
            run_index=run_index,
        )
        return opt_state, test_accuracy

    rng, initial_parameters = init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(initial_parameters)
    opt_state, test_accuracies = jax.lax.scan(
        epoch,
        init=opt_state,
        xs=jnp.arange(num_epochs),
        length=num_epochs,
    )
    jax.debug.print("Final test accuracy for run {}: {}", run_index, test_accuracies[-1])
    return opt_state, test_accuracies[-1]


def get_test_predictions_table(
    images: jax.Array, labels: jax.Array, outputs: jax.Array, predicted: jax.Array
):
    # ✨ W&B: Create a Table to store predictions for each test step
    test_table = wandb.Table(
        columns=["id", "image", "guess", "truth", *("score_" + str(i) for i in range(10))]
    )

    # convenience funtion to log predictions for a batch of test images
    def log_test_predictions(
        images: jax.Array,
        labels: jax.Array,
        outputs: jax.Array,
        predicted: jax.Array,
        test_table,
        log_counter,
    ):
        """Taken from these urls:
        - https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb#scrollTo=hgeXfCR575H0
        More specifically: https://wandb.ai/stacey/table-quickstart/runs/2ao7yrxx/code/_session_history.ipynb
        """
        # obtain confidence scores for all classes
        scores = jax.nn.softmax(outputs.data, 1)
        log_scores = np.asarray(scores)
        log_images = np.asarray(images)
        log_labels = np.asarray(labels)
        log_preds = np.asarray(predicted)
        # adding ids based on the order of the images
        _id = 0
        for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
            # add required info to data table:
            # id, image pixels, model's guess, true label, scores for all classes
            img_id = str(_id) + "_" + str(log_counter)
            test_table.add_data(img_id, wandb.Image(i), p, l, *s)
            _id += 1
            if _id == NUM_IMAGES_PER_BATCH:
                break


def loss(
    params: Params,
    inputs: jax.Array,
    targets: jax.Array,
    predict: Callable[[Params, jax.Array], jax.Array],
):
    preds = predict(params, inputs)
    loss_value = cross_entropy_loss(preds, targets)
    loss_value = jax.lax.pmean(loss_value, axis_name="batch")
    return loss_value, preds


def cross_entropy_loss(preds: jax.Array, targets: jax.Array):
    """Negative cross-entropy loss.

    Roughly: -1 * "How close the predicted probabilities match the target distribution"
    """
    return -jnp.mean(jnp.sum(preds * targets, axis=-1))


def get_accuracy(preds: jax.Array, targets: jax.Array):
    return jnp.mean(preds.argmax(-1) == targets.argmax(-1))


def mnist():
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = _mnist_raw()

    def _partial_flatten(x: np.ndarray) -> np.ndarray:
        """Flatten all but the first dimension of an ndarray."""
        return np.reshape(x, (x.shape[0], -1))

    def _one_hot(x: np.ndarray, k: int, dtype=np.float32) -> np.ndarray:
        """Create a one-hot encoding of x of size k."""
        return np.array(x[:, None] == np.arange(k), dtype)

    # train_images = _partial_flatten(train_images) / np.float32(255.0)
    # test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_images = train_images / np.float32(255.0)
    test_images = test_images / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


def _mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def _download(url, filename):
        """Download a url to a file in the JAX data temp directory."""
        if not path.exists(_DATA):
            os.makedirs(_DATA)
        out_file = path.join(_DATA, filename)
        if not path.isfile(out_file):
            urllib.request.urlretrieve(url, out_file)
            print(f"downloaded {url} to {_DATA}")

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))  # type: ignore
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))  # type: ignore
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


def _time_fn[**P, OutT](fn: Callable[P, OutT], desc: str = ""):
    desc = desc or fn.__name__

    @functools.wraps(fn)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> OutT:
        t0 = time.time()
        out = fn(*args, **kwargs)
        out = jax.block_until_ready(out)
        logger.info(f"`{desc}` took {time.time() - t0} seconds to complete.")
        return out

    return _wrapped


def setup_logging(local_rank: int, num_processes: int, verbose: int):
    if not sys.stdout.isatty():
        # Widen the log width when running in an sbatch script.
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
    )
    logger.setLevel(
        logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    )
    logging.getLogger("parallel_wandb").setLevel(
        logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    )
    logging.getLogger("jax").setLevel(
        logging.DEBUG if verbose == 3 else logging.INFO if verbose == 2 else logging.WARNING
    )


if __name__ == "__main__":
    main()
