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
    C = kernelmatrix(k, X_obs, X)
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
    old = 0
    r = m / n
    Cs = Matrix{eltype(X)}(undef, m, m)
    idxs = Vector{Int}(undef, m)

    for _ in 1:iters
        idxs = _sampleindex(X, r)
        C, Cs = _sample_matrix(k, X, idxs)
        old = _renyi_entropy(Cs, n, m)
        if old > best
            best = old
        end
    end

    return best, Cs, idxs
end

function factorization_entropy(svm::FixedSizeSVR, X, y)
    # Declare the size to work with
    n = size(y, 1)
    m = svm.subsample

    # Use this information to create the Nyström approximation
    kwargs = _kwargs2dict(svm)
    k = _choose_kernel(; kwargs...)
    (_, Cs, idxs) = _nystroem_renyi(k, X, n, m; iters=svm.iters)

    # We do a spectral decomposition
    fact = eigen(Cs)

    # We now augment the kernel matrix with ones to include the
    # bias term
    kern_mat_aug = hcat(fact.vectors, ones(m))

    # Build a named tuple for all the information
    info_tuple = (
        matrices=(X_matrix=X, eigen_fact=fact, aug_matrix=kern_mat_aug),
        target=(y_target=y, idxs=idxs),
    )

    return info_tuple
end

function svmtrain(svm::FixedSizeSVR, X, y)
    # Extract the necessary data from the arguments
    (X_matrix, fact, kern_mat_aug) = X
    (y_target, idxs) = y

    # Create a square matrix A^T * A from subsampled data
    sq_mat = kern_mat_aug' * kern_mat_aug

    # We need the correct size for the vector
    b = kern_mat_aug' * y_target[idxs]

    # We now solve the ridge regression problem, here we are using an iterative method
    λ_reg = 1.0 / svm.γ # The regularization parameter
    result, stats = cgls(sq_mat, b; λ=λ_reg)
    @assert check_if_solved(stats) == true # Always check if the iterative method converges

    # Extract the weights and the bias found
    weights = result[1:(end - 1)]
    bias = result[end]

    # Compute the alphas(i.e. weights) for the decision function
    alphas = svm.subsample .* fact.vectors
    @. alphas /= sqrt(fact.values)
    result = prod_reduction(alphas, weights)

    return X_matrix, result, bias, idxs
end

function svmpredict(svm::FixedSizeSVR, fits, xnew)
    (x, alphas, bias, idxs) = fits
    kwargs = _kwargs2dict(svm)
    k = _choose_kernel(; kwargs...)
    kern_mat = transpose(kernelmatrix(k, xnew, view(x, :, idxs)))
    alphas = dropdims(alphas; dims=1)
    result = prod_reduction(kern_mat, alphas) .+ bias

    # We need to remove the trailing dimension
    result = dropdims(result; dims=1)

    return result
end
