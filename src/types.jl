abstract type SVM end

mutable struct LSSVC{T} <: SVM
    x::AbstractMatrix{T}
    y::AbstractVector{T}
    α::AbstractVector{T}
    b::AbstractVector{T}
end

LSSVC(n::Int, m::Int) = LSSVC(zeros(n, m), zeros(m), zeros(m), zeros(m))
