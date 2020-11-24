"""
    SVM

A super type for both classifiers and regressors that are implemented as Support Vector Machines.
"""
abstract type SVM end

"""
    LSSVC()
    LSSVC(; kernel="rbf", γ=1.0, σ=1.0)

The type to hold a Least Squares Support Vector Classifier.

# Fields
- `kernel::String`: The kind of kernel to use for the non-linear mapping of the data.
- `γ::Float64`: The gamma hyperparameter that is intrinsic of the Least Squares version of the Support Vector Machines.
- `σ::Float64`: The hyperparameter for the RBF kernel.

# Keywords
- `kernel`: A string to denote the kernel to be used.
- `γ`: A float value to assign the gamma hyperparameter.
- `σ`: A float value to assign the sigma hyperparameter.
"""
mutable struct LSSVC <: SVM
    kernel::String
    γ::Float64
    σ::Float64
end

LSSVC(; kernel="rbf", γ=1.0, σ=1.0) = LSSVC(kernel, γ, σ)

@doc raw"""
    KernelRBF

This type is to compute the RBF kernel defined as

``K(x,y)=\exp{\left( -\vert x - y\vert^2 \gamma \right)}``

where ``\vert x - y\vert`` is the Euclidean norm. This norm is computed with the `Kernels.jl` package.

# Fields
- `γ::Real`: The hyperparameter associated with the RBF kernel.
"""
mutable struct KernelRBF
    γ::Real
end

"""
    KernelRBF(x::AbstractMatrix, gamma::Float64)
    KernelRBF(x, y::AbstractMatrix, gamma::Float64)
    KernelRBF(x, y::AbstractVector, gamma::Float64)

Depending on the arguments, it computes either a pairwise RBF kernel (if it is only with one matrix), or a pairwise RBF kernel between a matrix and an array.

# Arguments
- `x::AbstractMatrix`: A two-row matrix.
- `y::AbstractVector`: A one dimensional array.
- `gamma::Float64`: The hyperparameter needed to compute the kernel.
"""
function KernelRBF(x::AbstractMatrix, gamma::Float64)
    dist = SqEuclidean()
    r = pairwise(dist, x, dims=2)
    kernel = exp.(-r * gamma)

    return kernel
end

function KernelRBF(x, y::AbstractMatrix, gamma::Float64)
    dist = SqEuclidean()
    r = pairwise(dist, x, y, dims=2)
    kernel = exp.(-r * gamma)

    return kernel
end

function KernelRBF(x, y::AbstractVector, gamma::Float64)
    dist = SqEuclidean()
    r = colwise(dist, x, y)
    kernel = exp.(-r * gamma)

    return kernel
end
