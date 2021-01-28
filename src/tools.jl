"""
    An external product using Einstein summation.
"""
extern_prod(x, y) = @tullio z[i, j] := x[i] * y[j]

"""
    Pairwise multiplication when `x` and `a` both have two dimensions.
"""
pairwise_mul!(x, a) = @tullio x[i, j] *= a[i, j]

"""
    Pairwise difference when `x` and `y` are vectors.
"""
pairwise_diff(x, y) = @tullio z[i] := x[i] - y[i]

"""
    Computes a pairwise multiplication between `y`, `z` and the second dimension from
`x`. Then, it uses a sum reduction of the first dimension from `x`.

It is equivalent to

```julia
sum(@. x * y * z; dims=1)
```
"""
prod_reduction(x, y, z) = @tullio w[1, j] := x[i, j] * y[i] * z[i]
