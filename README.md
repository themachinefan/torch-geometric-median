# torch-geometric-mean

[![ci](https://img.shields.io/github/actions/workflow/status/chanind/torch-geometric-mean/ci.yaml?branch=main)](https://github.com/chanind/torch-geometric-mean)
[![PyPI](https://img.shields.io/pypi/v/torch-geometric-mean?color=blue)](https://pypi.org/project/torch-geometric-mean/)

A simplified version of the [geom-median](https://github.com/krishnap25/geom_median) Python library, updated to be higher performance on Pytorch and with full type-hinting. Thanks to [@themachinefan](https://github.com/themachinefan)!

## Installation

```
pip install torch-geometric-median
```

## Usage

This library exports a single function, `geometric_median`, which takes a tensor of shape `(N, D)` where `N` is the number of samples, and `D` is the size of each sample, and returns the geometric median of the points in the tensor .

```python
from torch_geometric_median import geometric_median

# Create a tensor of points
points = torch.tensor([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0],
    [4.0, 4.0],
])

# Compute the geometric median
median = geometric_median(points).median
```

### Backprop

Like the original [geom-median](https://github.com/krishnap25/geom_median) library, this library supports backpropagation through the geometric median computation.

```python
median = geometric_median(points).median
torch.linalg.norm(out.median).backward()
# The gradient of the median with respect to the input points is now in `points.grad`
```

### Extra options

The `geometric_median` function also supports a few extra options:

- `maxiter`: The maximum number of iterations to run the optimization for. Default is 100.
- `ftol`: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
- `weights`: A tensor of shape `(N,)` containing the weights for each point, where `N` is the number of samples. Default is `None`, which means all points are weighted equally.
- `show_progress`: If `True`, show a progress bar for the optimization. Default is `False`.
- `log_objective_values`: If `True`, log the objective value at each iteration under the key `objective_values_log`. Default is `False`.

```python
median = geometric_median(
    points,
    maxiter=1000,
    ftol=1e-10,
    weights=torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0]),
    show_progress=True,
    log_objective_values=True
).median
```

## Why does this library exist?

It appears that the original [geom-median](https://github.com/krishnap25/geom_median) library is no longer maintained, and [as pointed out by @themachinefan](https://github.com/jbloomAus/mats_sae_training/pull/22/files), the original library is not very performant on Pytorch. This library is a repackaging of [@themachinefan](https://github.com/themachinefan)'s improvements to the original geom-median library, simplying the code to just support pytorch, improving torch performance, and adding full type-hinting.

## Acknowledgements

This library is a repackaging of the work done by the original [geom-median](https://github.com/krishnap25/geom_median) library, and [@themachinefan](https://github.com/themachinefan) in their [PR](https://github.com/jbloomAus/mats_sae_training/pull/22/files), and as such, all credit goes to these incredible authors. If you use this library, you should cite [the original geom-median paper](https://ieeexplore.ieee.org/abstract/document/9721118).

## License

This library is licensed under a GPL license, as per the original geom-median library.

## Contributing

Contributions are welcome! Please open an issue or a PR if you have any suggestions or improvements. This library uses [PDM](https://pdm-project.org/) for dependency management, [Ruff](https://docs.astral.sh/ruff/) for linting, [Pyright](https://github.com/microsoft/pyright) for type-checking, and [Pytest](https://docs.pytest.org/en/8.0.x/) for tests.
