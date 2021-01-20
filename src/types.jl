"""
    SVM

A super type for both classifiers and regressors that are implemented as Support Vector Machines.
"""
abstract type SVM end

"""
    LSSVC()
    LSSVC(; kernel=:rbf, γ=1.0, σ=1.0, degree=0)

The type to hold a Least Squares Support Vector Classifier.

# Fields
- `kernel::Symbol`: The kind of kernel to use for the non-linear mapping of the data.
- `γ::Float64`: The gamma hyperparameter that is intrinsic of the Least Squares version of the Support Vector Machines.
- `σ::Float64`: The hyperparameter for the RBF kernel.
- `degree::Int`: The degree of the polynomial kernel. Only used if `kernel` is "poly".

"""
mutable struct LSSVC <: SVM
    kernel::Symbol
    γ::Float64
    σ::Float64
    degree::Int
end

LSSVC(; kernel=:rbf, γ=1.0, σ=1.0, degree::Int=0) = LSSVC(kernel, γ, σ, degree)

"""
    LSSVR()
    LSSVR(; kernel=:rbf, γ=1.0, σ=1.0, degree=0)

The type to hold a Least Squares Support Vector Regressor.

# Fields
- `kernel::Symbol`: The kind of kernel to use for the non-linear mapping of the data.
- `γ::Float64`: The gamma hyperparameter that is intrinsic of the Least Squares version of the Support Vector Machines.
- `σ::Float64`: The hyperparameter for the RBF kernel.
- `degree::Int`: The degree of the polynomial kernel. Only used if `kernel` is "poly".

"""
mutable struct LSSVR <: SVM
    kernel::Symbol
    γ::Float64
    σ::Float64
    degree::Int
end

LSSVR(; kernel=:rbf, γ=1.0, σ=1.0, degree::Int=0) = LSSVR(kernel, γ, σ, degree)
