# Elysivm

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://edwinb-ai.github.io/Elysivm/dev)
[![CI](https://github.com/edwinb-ai/Elysivm/workflows/CI/badge.svg)](https://github.com/edwinb-ai/Elysivm/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/edwinb-ai/Elysivm/branch/main/graph/badge.svg?token=U0HVBJ0ks7)](https://codecov.io/gh/edwinb-ai/Elysivm)

This is an implementation in pure `Julia` of the Least Squares Support Vector Machines [1] by Suykens and Vandewalle.
It contains both a **classifier** and a **regressor** implementation.

# MLJ Integration

It has been designed to be used with the fantastic [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/) Machine Learning framework.

# How to use it

It follows the same logic as the workflow endorsed by MLJ, with the only exception that the models here are based on Least Squares Support Vector Machines. If you need some guidance, check out the documentation where there are a couple of examples.


# Installation

You install this package from `Pkg` like so

```julia
Pkg> add https://github.com/edwinb-ai/Elysivm.git
```

This will install all the dependencies and you are good to go.

# References

[1. Least Squares Support Vector Machine Classifiers](https://link.springer.com/article/10.1023/A:1018628609742)