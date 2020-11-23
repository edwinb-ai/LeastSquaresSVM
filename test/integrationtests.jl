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
train, test = partition(eachindex(y), 2 / 3, shuffle=true, rng=15)
stand1 = Standardizer(count=true)
X = MLJBase.transform(MLJBase.fit!(MLJBase.machine(stand1, X)), X)


@testset "MLJ Integration" begin
    # Define a good set of hyperparameters for this problem
    model = Elysivm.LSSVClassifier(γ=80.0, σ=0.233333)
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
