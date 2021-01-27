extern_prod(x, y) = @tullio z[i, j] := x[i] * y[j]

pairwise_mul!(x, a) = @tullio x[i, j] *= a[i, j]
