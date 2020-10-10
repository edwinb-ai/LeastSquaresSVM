abstract type SVM end

mutable struct LSSVC{T} <: SVM
    x::AbstractMatrix{T}
    y::AbstractVector{T}
    α::AbstractVector{T}
    b::AbstractVector{T}
end

LSSVC(n::Integer, m::Integer) = LSSVC(zeros(n, m), zeros(m), zeros(m), zeros(m))

mutable struct KernelRBF
    γ::Real
end

function KernelRBF(x, gamma)
    dist = SqEuclidean()
    r = pairwise(dist, x, dims=2)
    kernel = exp.(-r * gamma)

    return kernel
end
