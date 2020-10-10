abstract type SVM end

mutable struct LSSVC{T} <: SVM
    x::AbstractMatrix{T}
    y::AbstractVector{T}
    α::AbstractVector{T}
    b::T
end

LSSVC(n::Integer, m::Integer) = LSSVC(zeros(n, m), zeros(m), zeros(m), 0.0)

mutable struct KernelRBF
    γ::Real
end

function KernelRBF(x, gamma)
    dist = SqEuclidean()
    r = pairwise(dist, x, dims=2)
    kernel = exp.(-r * gamma)

    return kernel
end

function KernelRBF(x, y, gamma)
    dist = SqEuclidean()
    r = colwise(dist, x, y)
    kernel = exp.(-r * gamma)

    return kernel
end
