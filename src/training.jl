function fit!(
    svm::LSSVC,
    x::AbstractMatrix,
    y::AbstractVector;
    kernel::String="rbf",
    params=(γ = 1.0, σ = 2.0),
)
    n = size(y, 1)
    # Initialize the necessary matrices
    Ω = build_omega(x, y; kernel=kernel, params=params)
    H = Ω + UniformScaling(1.0 / params.γ)

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

    # Save these values to the SVM object
    svm.b = b
    svm.α = α
    svm.y = y
    svm.x = x

    return nothing
end

function predict!(svm, x;kernel::String="rbf", params=(γ = 1.0, σ = 2.0))
    result = Vector{eltype(x)}(undef, size(x, 2))

    for i in axes(x, 2)
        kern_mat = KernelRBF(svm.x, x[:, i], params.σ)
        result[i] = sum(@. svm.y * kern_mat * svm.α) + svm.b
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
