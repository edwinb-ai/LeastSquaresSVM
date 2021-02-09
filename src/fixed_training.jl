function _sampleindex(X::AbstractMatrix, r::Real)
    0 < r <= 1 || throw(ArgumentError("Sample rate `r` must be in range (0,1]"))
    n = size(X, 2)
    m = ceil(Int, n * r)
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

    for _ = 1:iters
        idxs = sampleindex(X, r)
        C, Cs = sample_matrix(k, X, idxs)
        old = renyi_entropy(Cs, n, m)
        if old > best
            best = old
        end
    end

    return (best, Cs, idxs)
end

function svmtrain(svm::FixedSizeSVR, x::AbstractMatrix, y::AbstractVector)
    best, Cs, idxs = nystroem_renyi(k, x, n, m)
    fact = eigen(Cs)
    kern_mat_aug = hcat(fact.vectors, ones(m))
    sq_mat = kern_mat_aug' * kern_mat_aug
    b = kern_mat_aug' * y[idxs]
    result = (sq_mat + (I / gamma)) \ b

    wi_s = result[1:end-1]
    bias = result[end]

    alphas = m .* fact.vectors
    @. alphas /= sqrt(fact.values)
    result = sum(alphas .* wi_s; dims=1)

    return (result, bias, idxs)
end

function svmpredict(svm::FixedSizeSVR, x, y)
    alphas, bias, idxs = solve_problem(k, x, y, 0.001, n, m)
    kern_mat = kernelmatrix(k, view(x, :, idxs), x)
    estimates = sum(alphas * kern_mat; dims=1)
    preds = estimates .+ bias

    return preds
end
