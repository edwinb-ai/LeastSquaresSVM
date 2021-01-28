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

prod_reduction(x, y, z) = @tullio w[1, j] := x[i, j] * y[i] * z[i]
