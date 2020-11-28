# Elysivm

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://edwinb-ai.github.io/Elysivm/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://edwinb-ai.github.io/Elysivm/dev)
[![CI](https://github.com/edwinb-ai/Elysivm/workflows/CI/badge.svg)](https://github.com/edwinb-ai/Elysivm/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/edwinb-ai/Elysivm/branch/main/graph/badge.svg?token=U0HVBJ0ks7)](https://codecov.io/gh/edwinb-ai/Elysivm)

This is an implementation in pure `Julia` of the Least Squares Support Vector Machines [1] by Suykens and Vandewalle.

# MLJ Integration
It has been designed to be used with the fantastic [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/) Machine Learning framework. The implementation follows that of the `machine` type from MLJ. Some unit tests have been coded to ensure this integration.

# How to use it
It follows the same logic as the workflow endorsed by MLJ, with the only exception that the models is Least Squares Support Vector Machine. If you need some guidance, look at the [integration tests](https://github.com/edwinb-ai/Elysivm/blob/main/test/integrationtests.jl). Some example will be ready soon within the documentation.

You install this package from `Pkg` like so

```julia
Pkg> add https://github.com/edwinb-ai/Elysivm.git
```

This will install all the dependencies and you are good to go.

# References
[1. Least Squares Support Vector Machine Classifiers](https://link.springer.com/article/10.1023/A:1018628609742)