"""
    svmtrain(svm::LSSVC, x::AbstractMatrix, y::AbstractVector) -> Tuple

Solves a Least Squares Support Vector **classification** problem using the Conjugate
Gradient method. In particular, it uses the Lanczos process due to the fact that the
matrices are symmetric.

# Arguments
- `svm::LSSVC`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `x::AbstractMatrix`: The data matrix with the features. It is expected that this array is already standardized, i.e. the mean for each feature is zero and its standard deviation is one.
- `y::AbstractVector`: A vector that contains the classes. It is expected that there are only two classes, -1 and 1.

# Returns
- `Tuple`: A tuple containing `x`, `y` and the following two elements:
- `b`: Contains the bias for the decision function.
- `α`: Contains the weights for the decision function.
"""
function svmtrain(svm::LSSVC, x::AbstractMatrix, y::AbstractVector)
    n = size(y, 1)
    # Specify the keyword arguments
    kwargs = _kwargs2dict(svm)
    # We build the kernel matrix and the omega matrix
    kern_mat = _build_kernel_matrix(x; kwargs...)
    Ω = (y .* y') .* kern_mat
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

"""
    svmpredict(svm::LSSVC, fits, xnew::AbstractMatrix) -> AbstractArray

Uses the information obtained from `svmtrain` such as the bias and weights to construct a decision function and predict new class values. For the _classification_ problem only.

# Arguments
- `svm::LSSVC`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `fits`: It can be any container data structure but it must have four elements: `x`, the data matrix; `y`, the labels vector; `α`, the weights; and `b`, the bias.
- `xnew::AbstractMatrix`: The data matrix that contains the new instances to be predicted.

# Returns
- `Array`: The labels corresponding to the prediction to each of the instances in `xnew`.
"""
function svmpredict(svm::LSSVC, fits, xnew::AbstractMatrix)
    x, y, α, b = fits
    kwargs = _kwargs2dict(svm)
    @assert size(x, 1) == size(xnew, 1)
    kern_mat = _build_kernel_matrix(x, xnew; kwargs...)
    result = sum(@. kern_mat * y * α; dims=1) .+ b
    # We need to remove the trailing dimension
    result = reshape(result, size(result, 2))

    return sign.(result)
end

function svmtrain_mc(svm::LSSVC, x, y, nclass)
    num_classifiers = nclass * (nclass - 1) / 2
    # Create a collection to store the learned parameters
    class_parameters = Vector{Tuple}(undef, Int(num_classifiers))
    # Create a collection to store the class pairs
    class_pairs = Vector{Tuple}(undef, Int(num_classifiers))
    c_idx = 1 # For keeping track of the classifiers

    for idx in 1:nclass-1
        # Get the elements and indices for the first class
        a_class, a_idxs = _find_and_copy(idx, y)
        a_class .= 1.0 # The first class is encoded as 1.0
        for jdx in (idx + 1):nclass
            # Get the elements and indices for the second class
            b_class, b_idxs = _find_and_copy(idx, y)
            b_class .= -1.0 # The second class is encoded as -1.0

            # Train a binary classification problem with the first and second classes
            all_indxs = vcat(a_idxs, b_idxs) # Join all indices
            all_classes = vcat(a_class, b_class) # Join both classes

            # Always copy the model and use views for the samples
            samples = @view(x[:, all_indxs])
            # Solve the binary classification problem between classes idx and jdx
            # new_svm = deepcopy(svm)
            fits = svmtrain(svm, samples, all_classes)

            # We save the parameters of that classifier
            class_parameters[c_idx] = fits
            # We save the class pairs for decoding them later
            class_pairs[c_idx] = (idx, jdx)
            c_idx += 1
        end
    end

    return (deepcopy(svm), class_parameters, class_pairs)
end

function svmpredict_mc(fits, Xnew::AbstractMatrix)
    # Extract the model, parameters and class codes
    svm, params, pairs = fits
    nclass = length(fits[3]) # The third element are the class pairs
    # This will hold all the predictions for the classifiers
    pooled_predictions = Matrix{eltype(Xnew)}(undef, nclass, size(Xnew, 2))

    for (idx, p, s) in zip(1:nclass, params, pairs)
        # For prediction, we just need the fitted parameters
        prediction = svmpredict(svm, p, Xnew)

        # Now that we have prediction, decode them using the class pairs
        prediction[prediction .== 1.0] .= s[1]
        prediction[prediction .== -1.0] .= s[2]

        # Save the decoded predictions to our pooling collection
        pooled_predictions[idx, :] = prediction
    end

    # Apply the voting scheme
    results = _predictions_by_votes(pooled_predictions)

    return results
end

"""
    svmtrain(svm::LSSVR, x::AbstractMatrix, y::AbstractVector) -> Tuple

Solves a Least Squares Support Vector **regression** problem using the Conjugate Gradient
method. In particular, it uses the Lanczos process due to the fact that the matrices are
symmetric.

# Arguments
- `svm::LSSVR`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `x::AbstractMatrix`: The data matrix with the features. It is expected that this array is already standardized, i.e. the mean for each feature is zero and its standard deviation is one.
- `y::AbstractVector`: A vector that contains the continuous value of the function estimation. The elements can be any subtype of `Real`.

# Returns
- `Tuple`: A tuple containing `x` and the following two elements:
- `b`: Contains the bias for the decision function.
- `α`: Contains the weights for the decision function.
"""
function svmtrain(svm::LSSVR, x::AbstractMatrix, y::AbstractVector)
    n = size(y, 1)
    # Specify the keyword arguments
    kwargs = _kwargs2dict(svm)
    # We build the kernel matrix and the omega matrix
    kern_mat = _build_kernel_matrix(x; kwargs...)
    H = kern_mat + I / svm.γ

    # * Start solving the subproblems
    # First, solve for eta
    (η, stats) = cg_lanczos(H, ones(n))
    # Then, solve for nu
    (ν, stats) = cg_lanczos(H, y)

    # We then compute s
    s = dot(ones(n), η)

    # Finally, we solve the problem for alpha and b
    b = dot(η, y) / s
    α = ν .- (η * b)

    return (x, α, b)
end

"""
    svmpredict(svm::LSSVR, fits, xnew::AbstractMatrix) -> AbstractArray

Uses the information obtained from `svmtrain` such as the bias and weights to construct a
decision function and predict the new values of the function. For the _regression_
problem only.

# Arguments
- `svm::LSSVR`: The Support Vector Machine that contains the hyperparameters, as well as the kernel to be used.
- `fits`: It can be any container data structure but it must have four elements: `x`, the data matrix; `y`, the labels vector; `α`, the weights; and `b`, the bias.
- `xnew::AbstractMatrix`: The data matrix that contains the new instances to be predicted.

# Returns
- `Array`: The labels corresponding to the prediction to each of the instances in `xnew`.
"""
function svmpredict(svm::LSSVR, fits, xnew::AbstractMatrix)
    x, α, b = fits
    @assert size(x, 1) == size(xnew, 1)

    kwargs = _kwargs2dict(svm)
    kern_mat = _build_kernel_matrix(x, xnew; kwargs...)
    result = sum(kern_mat .* α; dims=1) .+ b

    # We need to remove the trailing dimension
    result = reshape(result, size(result, 2))

    return result
end
