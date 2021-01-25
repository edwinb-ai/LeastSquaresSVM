# LeastSquaresSVM

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://edwinb-ai.github.io/LeastSquaresSVM/dev)
[![CI](https://github.com/edwinb-ai/LeastSquaresSVM/workflows/CI/badge.svg)](https://github.com/edwinb-ai/LeastSquaresSVM/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/edwinb-ai/LeastSquaresSVM/branch/main/graph/badge.svg?token=U0HVBJ0ks7)](https://codecov.io/gh/edwinb-ai/LeastSquaresSVM)

This is an implementation in pure `Julia` of the Least Squares Support Vector Machines [1] by Suykens and Vandewalle.
It contains both a **classifier** and a **regressor** implementation.

# Installation

You install this package from `Pkg` like so

```julia
Pkg> add https://github.com/edwinb-ai/LeastSquaresSVM.git
```

This will install all the dependencies and you are good to go.

# References

[1. Least Squares Support Vector Machine Classifiers](https://link.springer.com/article/10.1023/A:1018628609742)