function svmtrain(svm::LSSVC, x::AbstractMatrix, y::AbstractVector)
    n = size(y, 1)
    # Initialize the necessary matrices
    Ω = build_omega(x, y, svm.σ; kernel=svm.kernel)
    H = Ω + I / svm.γ

    # * Start solving the subproblems
    # First, solve for eta
    (η, stats) = cg_lanczos(H, y)
    # Then, solve for nu
    (ν, stats) = cg_lanczos(H, ones(n))

    # We then compute s
    s = dot(y, η)

    # Finally, we solve the problem for alpha and b
    b = dot(η, ones(n)) / s
    α = ν .- (η * b)

    return (x, y, α, b)
end

function svmpredict(svm::LSSVC, fits, xnew::AbstractMatrix)
    x, y, α, b = fits
    # Compute the asymmetric kernel matrix in one go
    if svm.kernel == "rbf"
        kern_mat = KernelRBF(x, xnew, svm.σ)
    end
    result = sum(@. kern_mat * y * α; dims=1) .+ b
    # We need to remove the trailing dimension
    result = reshape(result, size(result, 2))

    return sign.(result)
end

function build_omega(
    x::AbstractMatrix,
    y::AbstractVector,
    sigma::Float64;
    kernel::String="rbf",
)
    if kernel == "rbf"
        # Compute using KernelFunctions
        kern_mat = KernelRBF(x, sigma)
        # Compute omega matrix
        Ω = (y .* y') .* kern_mat
    end

    return Ω
end
