##
## Types
##

MMI.@mlj_model mutable struct LSSVClassifier <: MMI.Deterministic
    kernel::String = "rbf"
    γ::Float64 = 1.0::(_ > 0.0)
    σ::Float64 = 1.0::(_ > 0.0)
end

MMI.@mlj_model mutable struct LSSVRegressor <: MMI.Deterministic
    kernel::String = "rbf"
    γ::Float64 = 1.0::(_ > 0.0)
    σ::Float64 = 1.0::(_ > 0.0)
end

##
## Fitting functions
##

function MMI.fit(model::LSSVClassifier, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X; transpose=true) # notice the transpose
    y_plain = convert(Array{eltype(Xmatrix)}, MMI.int(y))
    new_y = broadcast(x -> x == 2.0 ? -1.0 : 1.0, y_plain)
    decode  = MMI.decoder(y[1]) # for predict method

    cache = nothing

    svm = LSSVC(;kernel=model.kernel, γ=model.γ, σ=model.σ)
    fitted = svmtrain(svm, Xmatrix, new_y)

    fitresult = (deepcopy(svm), fitted, decode)

    report = (kernel = model.kernel, γ = model.γ, σ = model.σ)

    return (fitresult, cache, report)
end

function MMI.fit(model::LSSVRegressor, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X; transpose=true) # notice the transpose

    cache = nothing

    svr = LSSVR(;kernel=model.kernel, γ=model.γ, σ=model.σ)
    fitted = svmtrain(svr, Xmatrix, y)

    fitresult = (deepcopy(svr), fitted)

    report = (kernel = model.kernel, γ = model.γ, σ = model.σ)

    return (fitresult, cache, report)
end

##
## Prediction functions
##

function MMI.predict(model::LSSVClassifier, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew; transpose=true) # notice the transpose
    (svm, fitted, decode) = fitresult
    results = svmpredict(svm, fitted, Xmatrix)
    results = broadcast(x -> x == -1.0 ? 2.0 : 1.0, results)
    y = convert(Array{UInt64}, results)

    return decode(y)
end

function MMI.predict(model::LSSVRegressor, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew; transpose=true) # notice the transpose
    (svr, fitted) = fitresult
    results = svmpredict(svr, fitted, Xmatrix)

    return results
end