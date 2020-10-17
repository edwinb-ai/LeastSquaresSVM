abstract type SVM end

mutable struct LSSVC <: SVM
    kernel::String
    γ::Float64
    σ::Float64
end

LSSVC(; kernel="rbf", γ=1.0, σ=1.0) = LSSVC(kernel, γ, σ)

mutable struct KernelRBF
    γ::Real
end

function KernelRBF(x, gamma)
    dist = SqEuclidean()
    r = pairwise(dist, x, dims=2)
    kernel = exp.(-r * gamma)

    return kernel
end

function KernelRBF(x, y::AbstractMatrix, gamma)
    dist = SqEuclidean()
    r = pairwise(dist, x, y, dims=2)
    kernel = exp.(-r * gamma)

    return kernel
end

function KernelRBF(x, y::AbstractVector, gamma)
    dist = SqEuclidean()
    r = colwise(dist, x, y)
    kernel = exp.(-r * gamma)

    return kernel
end
