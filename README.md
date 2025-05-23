# parallel_wandb

This simple package shows how to use the new feature of Weights & Biases (wandb)
for parallel runs: [`reinit="create_new"`](https://docs.wandb.ai/guides/runs/multiple-runs-per-process/#example-concurrent-processes).

- It provides simple functions to initialize multiple wandb runs and log data to them in parallel.
- The `wandb_log` function can also be used in combination with `jax.vmap` for highly efficient training of multiple models at once, now with Wandb logging!

A demonstration of this can be found in `jax_mnist.py`.

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
