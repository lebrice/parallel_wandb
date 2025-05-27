# parallel_wandb

This simple package makes it easy to use the new ([`reinit="create_new"`](https://docs.wandb.ai/guides/runs/multiple-runs-per-process/#example-concurrent-processes))
feature of Weights & Biases (wandb) to create and log to multiple wandb runs in parallel

This, when combined with `jax.vmap`, enables extremely efficient, high-throughput training (**and logging**!) of multiple simultaneous training runs.

- This package provides two simple functions that you can import and use in your own project: `wandb_init` to initialize multiple wandb runs and `wandb_log` to log metrics to them in parallel.
- A demonstration of how these can be used with jax.vmap can be found in `jax_mnist.py`.

## Installation

1. (optional) Install UV: https://docs.astral.sh/uv/getting-started/installation/

2. Add this package as a dependency to your project:

```console
uv add parallel_wandb
```

OR, if you don't use UV yet, you can also `pip install parallel_wandb`.

## Usage

```python
from parallel_wandb import wandb_init, wandb_log

runs = wandb_init(
    {"name": ["run_0", "run_1"], "config": {"seed": [0, 1]}},
    project="test_project",
    name="test_name",
)
assert isinstance(runs, np.ndarray) and runs.shape == (2,) and runs.dtype == object

wandb_log(runs, {"loss": [0.1, 0.2]}, step=0)
```
