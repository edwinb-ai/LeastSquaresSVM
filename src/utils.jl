"""
    This revert function reverses the internal implementation of the
`ScaleTransform` operation from `KernelFunctions.jl`. This is to preserve the
previous convention of this package.
"""
_revert(x) = sqrt(2.0 * x)

"""
    _build_kernel_matrix(x; kwargs...) -> AbstractMatrix
    _build_kernel_matrix(x, y; kwargs...) -> AbstractMatrix

It builds a kernel matrix using a user specified kernel. When just one matrix is passed,
`x`, the resulting matrix has the same size as `x`.
When both `x` and `y` are passed, the resulting matrix has the size
(`size(x, 2)`, `size(y, 2)`).

The matrix is built using the `kernelmatrix!` functions from `KernelFunctions.jl`.

# Arguments
- `x::AbstractMatrix`: The data matrix with the training instances.
- `y::AbstractMatrix`: Another data matrix, normally used for the prediction step in the classification implementation.

# Keywords
- `kernel::String="rbf"`: The kernel to be used. The available options are "rbf" for a RBF type kernel; "linear", for a classic linear kernel; and "poly" for a polynomial kernel.
- `sigma::Float64`: The hyperparameter for the RBF kernel.
- `degree::Float64`: The degree of the polynomial kernel if "poly" is chosen.

# Returns
- `kern_mat`: The kernel matrix.
"""
function _build_kernel_matrix(x; kwargs...)
    # Build the result matrix
    # kern_mat = similar(x)

    # Extract the matrix for the keyword arguments
    kernel = kwargs[:kernel]

    if kernel == "rbf"
        # Create the kernel with the corresponding scale
        t = ScaleTransform(_revert(kwargs[:sigma]))
        κ = transform(SqExponentialKernel(), t)
        kern_mat = kernelmatrix(κ, x)
    elseif kernel == "linear"
        κ = LinearKernel()
        kern_mat = kernelmatrix(κ, x)
    elseif kernel == "poly"
        # Create the kernel with the corresponding degree
        κ = PolynomialKernel(kwargs[:degree], 0.0)
        kern_mat = kernelmatrix(κ, x)
    end

    return kern_mat
end

function _build_kernel_matrix(x, y; kwargs...)
    # Check that the first dimension is the same
    @assert size(x, 1) == size(y, 1)

    # Extract the second dimension to build the result matrix
    # n = size(x, 2)
    # m = size(y, 2)
    # kern_mat = Matrix{eltype(x)}(undef, n, m)

    # Extract the matrix for the keyword arguments
    kernel = kwargs[:kernel]

    if kernel == "rbf"
        # Create the kernel with the corresponding scale
        t = ScaleTransform(_revert(kwargs[:sigma]))
        κ = transform(SqExponentialKernel(), t)
        kern_mat = kernelmatrix(κ, x, y)
    elseif kernel == "linear"
        κ = LinearKernel()
        kern_mat = kernelmatrix(κ, x, y)
    elseif kernel == "poly"
        # Create the kernel with the corresponding degree
        κ = PolynomialKernel(kwargs[:degree], 0.0)
        kern_mat = kernelmatrix(κ, x, y)
    end

    return kern_mat
end
