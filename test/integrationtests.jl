using DataFrames, CSV
using CategoricalArrays
using Random
using MLJ

# @testset "MLJ Integration Classification" begin
#     # ============== Problem setup ============ #
#     # This is a binary classification problem
#     headers = [
#         "id", "Clump Thickness",
#         "Uniformity of Cell Size", "Uniformity of Cell Shape",
#         "Marginal Adhesion", "Single Epithelial Cell Size",
#         "Bare Nuclei", "Bland Chromatin",
#         "Normal Nucleoli", "Mitoses", "class"
#     ]

#     path = "wbc.csv"
#     # Replace the "?" to `missing`
#     data = DataFrame(CSV.File(path; header=headers, missingstring="?"))

#     # Don't include the id's
#     select!(data, Not(:id))

#     # Transform to categorical
#     transform!(data, :class => categorical, renamecols=false)

#     # We don't need `missing`'s
#     data = dropmissing(data)

#     # We change the integer types to floating point types, and multiclass for the
#     # classes
#     coerce!(data, Count => Continuous, :class => Multiclass)

#     # Split the training and test data
#     y, X = unpack(data, ==(:class), colname -> true)
#     train, test = partition(eachindex(y), 2.0 / 3.0, shuffle=true, rng=20)

#     # Define a good set of hyperparameters for this problem
#     pipe = MLJ.@pipeline(MLJ.Standardizer(), LSSVClassifier(γ=80.0, σ=0.233333))
#     mach = MLJ.machine(pipe, X, y)
#     MLJ.fit!(mach, rows=train, verbosity=0)

#     results = MLJ.predict(mach, rows=test)
#     acc = MLJ.accuracy(results, y[test])

#     # Test for correctness
#     @test isreal(acc)
#     # Test for accuracy, at least 95% for this problem
#     @test acc >= 0.95
# end

# @testset "MLJ Integration Regressor" begin
#     # We create a dataset of 100 samples and 5 features, with some gaussian noise
#     X, y = MLJ.make_regression(100, 5; noise=0.5, rng=18)
#     df = DataFrame(X)
#     df.y = y
#     dfnew = coerce(df, autotype(df))

#     y, X = unpack(dfnew, ==(:y), colname -> true)
#     train, test = partition(eachindex(y), 0.75, shuffle=true, rng=20)

#     # Define a good set of hyperparameters for this problem
#     model = MLJ.@pipeline(MLJ.Standardizer(), LSSVRegressor(γ=10.0, σ=0.5))
#     mach = MLJ.machine(model, X, y)
#     MLJ.fit!(mach, rows=train, verbosity=0)

#     ŷ = MLJ.predict(mach, rows=test)
#     result = round(MLJ.rms(ŷ, y[test]), sigdigits=4)
#     @show result

#     # Test for correctness
#     @test isreal(result)
# end

@testset "MLJ Integration Fixed Size Regressor" begin
    n = 500 # Total number of points or samples
    m = 10 # Subset of points or samples
    X, y = MLJ.make_regression(n, 2; noise=0.5, rng=18) # Here, 2 is the number of features
    df = DataFrame(X)
    df.y = y
    dfnew = coerce(df, autotype(df))

    y, X = unpack(dfnew, ==(:y), colname -> true)
    train, test = partition(eachindex(y), 0.7, shuffle=true, rng=20)
    # Center the data
    stand1 = MLJ.Standardizer()
    X = MLJ.transform(MLJ.fit!(MLJ.machine(stand1, X)), X)

    # Define some hyperparameters for the Fixed Size regressor
    fixed_svr = FixedSizeRegressor(γ=6.0, σ=4500, subsample=m)
    mach = MLJ.machine(fixed_svr, X, y)
    MLJ.fit!(mach, rows=train, verbosity=0)
    ŷ = MLJ.predict(mach, rows=test)
    result = round(MLJ.rms(ŷ, y[test]), sigdigits=4)
    println("Fixed size")
    @show result

    # Repeat the problem with a baseline regressor
    svr = LSSVRegressor(γ=6.0, σ=4500)
    mach = MLJ.machine(svr, X, y)
    MLJ.fit!(mach, rows=train, verbosity=0)
    ŷ = MLJ.predict(mach, rows=test)
    result = round(MLJ.rms(ŷ, y[test]), sigdigits=4)
    println("Baseline")
    @show result

    # Test for correctness
    @test isreal(result)
end

# @testset "Multiclass classification" begin
#     X, y = MLJ.@load_iris
#     train, test = partition(eachindex(y), 0.6, shuffle=true, rng=30)
#     pipe = MLJ.@pipeline(MLJ.Standardizer(), LSSVClassifier(γ=80.0, σ=0.1))
#     mach = MLJ.machine(pipe, X, y)
#     MLJ.fit!(mach, rows=train, verbosity=0)
#     results = MLJ.predict(mach, rows=test)
#     acc = MLJ.accuracy(results, y[test])

#     # Check that it is not NaN, and never zero
#     @test isreal(acc) && acc >= 0.9
# end
