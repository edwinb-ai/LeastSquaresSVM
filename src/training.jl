    function fit!(
    svm::LSSVC,
    x::AbstractMatrix,
    y::AbstractVector;
    kernel::String="rbf",
    params=(γ = 1.0, σ = 2.0),
)
    Ω = _build_omega(x, y; kernel=kernel, params=params)

    return Ω
end

function _build_omega(
    x::AbstractMatrix,
    y::AbstractVector;
    kernel::String="rbf",
    params=(γ = 1.0, σ = 2.0),
)
    n = size(x, 2)

    if kernel == "rbf"
        prefactor = params.σ
        # Compute using KernelFunctions
        kern_mat = KernelRBF(x, prefactor)
        # Remove the diagonal factors
        # Compute omega matrix
        Ω = (y * y') .* kern_mat
    end

    return Ω
end
