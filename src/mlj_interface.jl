MMI.@mlj_model mutable struct LSSVClassifier <: MMI.Deterministic
    kernel::String = "rbf"
    γ::Float64 = 1.0::(_ > 0.0)
    σ::Float64 = 1.0::(_ > 0.0)
end

function MMI.fit(model::LSSVClassifier, verbosity::Int, X, y)
    Xmatrix = adjoint(MMI.matrix(X)) # notice the transpose
    y_plain = MMI.int(y)
    decode  = MMI.decoder(y[1]) # for predict method

    cache = nothing

    svm = LSSVC()
    fit!(svm, Xmatrix, y;kernel=model.kernel, params=(γ = model.γ, σ = model.σ))

    fitresult = (deepcopy(svm), decode)

    return (fitresult, cache, report)
end

function MMI.predict(model::LSSVClassifier, fitresult, Xnew)
    (svm, decode) = fitresult
    results = predict!(svm, Xnew;kernel=model.kernel, params=(γ = model.γ, σ = model.σ))

    return decode(results)
end
