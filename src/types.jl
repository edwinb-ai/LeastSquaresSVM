mutable struct LSSVM
    x::AbstractArray
    y::AbstractArray
    α::AbstractArray
    b::AbstractArray
end

LSVM(n::Int, m::Int) = LSVM(zeros(n, m), zeros(m), zeros(m), zeros(m))
