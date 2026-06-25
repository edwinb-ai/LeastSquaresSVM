using DataFrames, CSV
using CategoricalArrays

@testset "MLJ Integration Classification" begin
    # ============== Problem setup ============ #
    headers = [
        "id", "Clump Thickness",
        "Uniformity of Cell Size", "Uniformity of Cell Shape",
        "Marginal Adhesion", "Single Epithelial Cell Size",
        "Bare Nuclei", "Bland Chromatin",
        "Normal Nucleoli", "Mitoses", "class"
    ]

    path = "wbc.csv"
    # Replace the "?" to `missing`
    data = CSV.File(path; header=headers, missingstring="?") |> DataFrame

    # Don't include the id's
    select!(data, Not(:id))

    # Change the class tags
    replace!(data.class, 2 => -1)
    replace!(data.class, 4 => 1)

    # Transform to categorical
    transform!(data, :class => categorical, renamecols=false)

    # We don't need `missing`'s
    data = dropmissing(data)

    # Split the training and test data
    y, X = unpack(data, ==(:class), colname -> true)
    train, test = partition(eachindex(y), 2 / 3, shuffle=true, rng=15)

    # Define a good set of hyperparameters for this problem
    pipe = MLJ.Pipeline(Standardizer(count=true), LSSVClassifier(γ=80.0, σ=0.233333))
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, rows=train, verbosity=0)

    results = MLJ.predict(mach, rows=test)
    acc = MLJ.accuracy(results, y[test])

    # Test for correctness
    @test isreal(acc)
    # Test for accuracy, at least 95% for this problem
    @test acc >= 0.95
end

@testset "MLJ Integration Regressor" begin
    X, y = MLJ.make_regression(100, 5; noise=0.5, rng=18)
    df = DataFrame(X)
    df.y = y
    dfnew = coerce(df, autotype(df))

    y, X = unpack(dfnew, ==(:y), colname -> true)
    train, test = partition(eachindex(y), 0.75, shuffle=true, rng=20)

    # Define a good set of hyperparameters for this problem
    model = MLJ.Pipeline(Standardizer(), LSSVRegressor(γ=10.0, σ=0.5))
    mach = MLJ.machine(model, X, y)
    MLJ.fit!(mach, rows=train, verbosity=0)

    ŷ = MLJ.predict(mach, rows=test)
    result = round(MLJ.rms(ŷ, y[test]), sigdigits=4)

    # Test for correctness
    @test isreal(result)
end

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
    ŷ = MLJ.predict(mach, rows=test)
    result = round(MLJ.rms(ŷ, y[test]), sigdigits=4)
    println("Fixed size")
    @show result

    # Repeat the problem with a baseline regressor
    svr = LSSVRegressor(γ=6.0, σ=4500)
    mach = MLJ.machine(svr, X, y)
    MLJ.fit!(mach, rows=train, verbosity=0)
    ŷ = MLJ.predict(mach, rows=test)
    result = round(MLJ.rms(ŷ, y[test]), sigdigits=4)
    println("Baseline")
    @show result

    # Test for correctness
    @test isreal(result)
end

@testset "Multiclass classification" begin
    X, y = MLJ.@load_iris
    train, test = partition(eachindex(y), 0.6, shuffle=true, rng=30)
    pipe = MLJ.Pipeline(Standardizer(), LSSVClassifier(γ=80.0, σ=0.1))
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, rows=train, verbosity=0)
    results = MLJ.predict(mach, rows=test)
    acc = MLJ.accuracy(results, y[test])

    # Check that it is not NaN, and never zero
    @test isreal(acc) && acc >= 0.9
end
