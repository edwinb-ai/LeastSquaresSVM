function fit!(svm::LSSVC, x::AbstractMatrix, y::AbstractVector)
    n = size(y, 1)
    # Initialize the necessary matrices
    Ω = build_omega(x, y; kernel=svm.kernel, params=(γ = svm.γ, σ = svm.σ))
    H = Ω + UniformScaling(1.0 / svm.γ)

    # * Start solving the subproblems

    # First, solve for eta
    η, stats = cg(H, y)
    # Then, solve for nu
    ν, stats = cg(H, ones(n))

    # We then compute s
    s = dot(y, η)

    # Finally, we solve the problem for alpha and b
    b = dot(η, ones(n) / s)
    α = ν .- (η * b)

    return (x, y, α, b)
end

function predict!(svm::LSSVC, fits, xnew::AbstractMatrix)
    result = Vector{eltype(xnew)}(undef, size(xnew, 2))

    x, y, α, b = fits

    for i in axes(xnew, 2)
        if svm.kernel == "rbf"
            kern_mat = KernelRBF(x, xnew[:, i], svm.σ)
        end
        result[i] = sum(@. y * kern_mat * α) + b
    end

    return sign.(result)
end

function build_omega(
    x::AbstractMatrix,
    y::AbstractVector;
    kernel::String="rbf",
    params=(γ = 1.0, σ = 2.0),
)
    if kernel == "rbf"
        # Compute using KernelFunctions
        kern_mat = KernelRBF(x, params.σ)
        # Compute omega matrix
        Ω = (y * y') .* kern_mat
    end

    return Ω
end
