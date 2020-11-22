using DataFrames, CSV
using CategoricalArrays
import Elysivm

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

# Separar los conjuntos de entrenamiento y prueba
y, X = unpack(data, ==(:class), colname -> true)
train, test = partition(eachindex(y), 0.67, shuffle=true, rng=11)
stand1 = Standardizer()
X = MLJBase.transform(MLJBase.fit!(MLJBase.machine(stand1, X)), X)

# Mostrar estadísticas
display(describe(X))

@testset "MLJ Integration" begin
    model = Elysivm.LSSVClassifier()
    r1 = range(model, :γ; lower=0.0001, upper=0.001)
    r2 = range(model, :σ; lower=0.0001, upper=0.001)
    self_tuning_model = TunedModel(
        model=model,
        tuning=Grid(goal=500),
        resampling=StratifiedCV(nfolds=5),
        range=[r1, r2],
        measure=MLJBase.accuracy,
        # acceleration=CPU1(),
        acceleration=CPUThreads(),
    )
    # mach = MLJ.machine(model, X, y)
    mach = MLJBase.machine(self_tuning_model, X, y)
    MLJBase.fit!(mach, rows=train)
    # display(report(mach))
    display(fitted_params(mach).best_model)

    results = MLJBase.predict(mach, rows=test)
    acc = MLJBase.accuracy(results, y[test])
    display(acc)

    # Don't test for correctness, test that is works
    @test isreal(acc)
end
