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
    stand1 = Standardizer(count=true)
    X = MLJBase.transform(MLJBase.fit!(MLJBase.machine(stand1, X)), X)

    # Define a good set of hyperparameters for this problem
    model = LSSVClassifier(γ=80.0, σ=0.233333)
    mach = MLJ.machine(model, X, y)
    MLJBase.fit!(mach, rows=train)

    results = MLJBase.predict(mach, rows=test)
    acc = MLJBase.accuracy(results, y[test])
    display(acc)

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
    display(first(dfnew, 3) |> pretty)
    display(describe(dfnew, :mean, :std, :eltype))

    y, X = unpack(dfnew, ==(:y), colname -> true)
    train, test = partition(eachindex(y), 0.75, shuffle=true, rng=20)
    stand1 = Standardizer()
    X = MLJBase.transform(MLJBase.fit!(MLJBase.machine(stand1, X)), X)
    display(describe(X |> DataFrame, :mean, :std, :eltype))

    # Define a good set of hyperparameters for this problem
    model = LSSVRegressor(γ=10.0, σ=0.5)
    mach = MLJ.machine(model, X, y)
    MLJBase.fit!(mach, rows=train)

    ŷ = MLJBase.predict(mach, rows=test)
    result = round(MLJBase.rms(ŷ, y[test]), sigdigits=4)
    display(result)

    # Test for correctness
    @test isreal(result)
end

@testset "Multiclass classification" begin
    X, y = @load_iris
    train, test = partition(eachindex(y), 0.6, shuffle=true, rng=30)
    pipe = @pipeline(Standardizer(), LSSVClassifier(γ=80.0, σ=0.1))
    mach = MLJBase.machine(pipe, X, y)
    MLJBase.fit!(mach, rows=train)
    results = MLJBase.predict(mach, rows=test)
    acc = MLJBase.accuracy(results, y[test])
    @show acc

    # Check that it is not NaN, and never zero
    @test isreal(acc) && acc >= 0.9
end
