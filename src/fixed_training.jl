function _sampleindex(X::AbstractMatrix, r::Real)
    # Check that `r` is always within the open interval (0, 1]
    0 < r <= 1 || throw(ArgumentError("Sample rate `r` must be in range (0,1]"))
    n = size(X, 2)
    m = ceil(Int, n * r)
    # We sample indices **without** replacement
    S = StatsBase.sample(1:n, m; replace=false, ordered=true)

    return S
end

function _sample_matrix(k::Kernel, X::AbstractMatrix, S::Vector{<:Integer})
    X_obs = view(X, :, S)
    C = kernelmatrix(k, X_obs, X; obsdim=2)
    Cs = C[:, S]

    return (C, Cs)
end

function _renyi_entropy(X::AbstractMatrix, N::Integer, M::Integer)
    ones_m = ones(M)
    integral = BL.gemv('N', 1.0, X, ones_m)
    entropy = BL.dot(M, ones_m, 1, integral, 1)
    entropy /= N^2

    return -log(entropy)
end

function _nystroem_renyi(k::Kernel, X::AbstractMatrix, n, m; iters=50_000)
    @assert m < n

    best = -Inf
    r = m / n
    best_Cs = Matrix{eltype(X)}(undef, m, m)
    best_idxs = Vector{Int}(undef, m)

    for _ in 1:iters
        idxs = _sampleindex(X, r)
        _, Cs = _sample_matrix(k, X, idxs)
        ent = _renyi_entropy(Cs, n, m)
        if ent > best
            best = ent
            best_Cs = Cs
            best_idxs = idxs
        end
    end

    return best, best_Cs, best_idxs
end

function factorization_entropy(svm::FixedSizeSVR, X, y)
    # Declare the size to work with
    n = size(y, 1)
    m = svm.subsample

    # Actively select a working set of `m` prototype (support) vectors and obtain
    # the m × m kernel matrix of that working set for the Nyström approximation.
    kwargs = _kwargs2dict(svm)
    k = _choose_kernel(; kwargs...)
    (_, K_mm, idxs) = _nystroem_renyi(k, X, n, m; iters=svm.iters)

    # Spectral decomposition of the (symmetric) prototype kernel matrix.
    fact = eigen(Symmetric(K_mm))

    # Build a named tuple for all the information
    info_tuple = (
        matrices=(X_matrix=X, eigen_fact=fact),
        target=(y_target=y, idxs=idxs),
    )

    return info_tuple
end

"""
    _nystrom_map(fact) -> P

Build the Nyström projection `P = U Λ^{-1/2}` from the eigendecomposition `fact`
of the m × m prototype kernel matrix, keeping only the numerically significant
eigenpairs (so a point's approximate feature vector is `φ̂(x) = (k_m(x))ᵀ P`,
following eq. (6.12) of Suykens et al.). Truncating tiny eigenvalues avoids the
`1/√λ` blow-up.
"""
function _nystrom_map(fact)
    λ = fact.values
    U = fact.vectors
    tol = sqrt(eps(float(eltype(λ)))) * maximum(λ)
    keep = λ .> tol
    return U[:, keep] ./ sqrt.(λ[keep])'
end

function svmtrain(svm::FixedSizeSVR, X, y)
    # Extract the necessary data from the arguments
    (X_matrix, fact) = X
    (y_target, idxs) = y
    n = size(X_matrix, 2)

    # Nyström feature map and the feature matrix Φ for ALL n training points:
    # Φ = K_nm * P  (size n × r), with K_nm the kernel between every training
    # point and the m prototypes.
    P = _nystrom_map(fact)
    kwargs = _kwargs2dict(svm)
    k = _choose_kernel(; kwargs...)
    prototypes = view(X_matrix, :, idxs)
    K_nm = kernelmatrix(k, X_matrix, prototypes; obsdim=2)
    Φ = K_nm * P
    r = size(Φ, 2)

    # Primal ridge regression over all n points (Suykens et al., eq. 6.11):
    #   min_{w,b} (1/2)‖w‖² + (γ/2) Σ_{k=1}^n (y_k - wᵀφ̂(x_k) - b)².
    # Solve the regularized normal equations; the 1/γ ridge applies to w only,
    # not to the bias term.
    A = hcat(Φ, ones(eltype(Φ), n))
    H = A' * A
    @inbounds for i in 1:r
        H[i, i] += 1.0 / svm.γ
    end
    θ = H \ (A' * y_target)
    weights = θ[1:r]
    bias = θ[end]

    # Dual support values for the decision function: α = P w  (eq. 6.14), so the
    # model is y(x) = Σ_k α_k K(x_k, x) + b.
    alphas = P * weights

    return X_matrix, alphas, bias, idxs
end

function svmpredict(svm::FixedSizeSVR, fits, xnew)
    (x, alphas, bias, idxs) = fits
    kwargs = _kwargs2dict(svm)
    k = _choose_kernel(; kwargs...)
    prototypes = view(x, :, idxs)
    kern_mat = kernelmatrix(k, xnew, prototypes; obsdim=2)  # (n_new × m)

    return kern_mat * alphas .+ bias
end
