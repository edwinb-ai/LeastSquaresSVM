using DataFrames, CSV
using CategoricalArrays
import Elysivm

# ============== Problem setup ============ #
headers = [
	"id", "class",
	"mean radius", "mean texture",
	"mean perimeter", "mean area",
	"mean smoothness", "mean compactness",
	"mean concavity", "mean concave points",
	"mean symmetry", "mean fractal dimension",
	"radius error", "texture error",
	"perimeter error", "area error",
	"smoothness error", "compactness error",
	"concavity error", "concave points error",
	"symmetry error", "fractal dimension error",
	"worst radius", "worst texture",
	"worst perimeter", "worst area",
	"worst smoothness", "worst compactness",
	"worst concavity", "worst concave points",
	"worst symmetry", "worst fractal dimension"
]

# Load data into DataFrame
path = "wdbc.csv"
data = CSV.File(path; header=headers) |> DataFrame

# Do not include id's
select!(data, Not(:id))

categorical!(data, :class)

# Split into training and testing sets
y, X = unpack(data, ==(:class), colname -> true)
train, test = partition(eachindex(y), 0.80, shuffle=true, rng=11)

@testset "MLJ Integration" begin
    model = Elysivm.LSSVClassifier()
    r1 = range(model, :γ, lower=0.0001, upper=0.1)
    r2 = range(model, :σ, lower=0.1, upper=2.0)
    self_tuning_model = TunedModel(
        model=model,
        tuning=Grid(goal=100),
        resampling=CV(nfolds=5),
        range=[r1, r2],
        measure=accuracy,
    )
    # machine = MLJ.machine(model, X, y)
    machine = MLJBase.machine(self_tuning_model, X, y)
    MLJBase.fit!(machine; rows=train)

    results = MLJBase.predict(machine; rows=test)
    acc = MLJBase.accuracy(results, y[test])
    display(acc)

    # Don't test for correctness, test that is works
    @test isreal(acc)
end
