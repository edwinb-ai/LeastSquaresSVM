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
    H = Ω + UniformScaling(params.γ)

    # * Start solving the problems

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
    kern_mat = KernelRBF(svm.x, x, params.σ)
    result = @. svm.y * kern_mat * svm.α

    return sign(sum(result) + svm.b)
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
